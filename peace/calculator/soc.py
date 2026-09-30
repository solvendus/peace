import numpy as np
import jax
import jax.numpy as jnp
from peace import units
from peace.nn.io import load_model_artifacts,atomic_energy_offset
from peace.nn.soc import soc_quantities,spin_free_pairs,spin_root_indices
from peace.graph import AtomicNumberTable
from peace.runtime import model_jit
from peace.calculator.calculator import PEACECalculator
from peace.calculator.phase import DEFAULT_NAC_DOT_THRESHOLD

def _expand_sf_np(a,ns,nt):
    out=np.zeros((ns+3*nt,ns+3*nt)+a.shape[2:],dtype=a.dtype)
    out[:ns,:ns]=a[:ns,:ns]
    for m in range(3):
        sl=slice(ns+m*nt,ns+(m+1)*nt)
        out[sl,sl]=a[ns:,ns:]
    return out

class SOCCalculator(PEACECalculator):
    def __init__(self,modelpath,paramspath,atom_types,
                 warn_on_state_mixing=True,
                 phase_dot_threshold=DEFAULT_NAC_DOT_THRESHOLD):
        model,params,cfg=load_model_artifacts(modelpath,paramspath)
        if cfg.get('include_soc') is not True:
            raise ValueError('SOCCalculator requires explicit include_soc: true')
        self.model_configs=cfg
        self.ns,self.nt=cfg['n_singlets'],cfg['n_triplets']
        self.n_sf=self.ns+self.nt
        self.n_total_states=self.ns+3*self.nt
        self.ztable=AtomicNumberTable.from_dict(cfg['mapping'])
        self.use_au=cfg['use_au'];self.cutoff=cfg['r_max']
        self.atom_types=np.asarray(atom_types,dtype=np.int32)
        self.n_atoms=len(self.atom_types)
        self.energy_offset=atomic_energy_offset(cfg,atom_types)
        self.has_shared_smooth_cutoff=True
        self.has_rigid_connection=True
        self.warn_on_state_mixing=bool(warn_on_state_mixing)
        self.phase_dot_threshold=float(phase_dot_threshold)
        if not np.isfinite(self.phase_dot_threshold) or self.phase_dot_threshold<0:
            raise ValueError('phase_dot_threshold must be finite and non-negative')
        self.nac_idx=tuple(np.asarray(spin_free_pairs(self.ns,self.nt)).T)
        self.safe_threshold=1.e-9 if self.use_au else 1.e-7
        self.reset_phase_tracking()
        self._raw=lambda g:soc_quantities(model,params,g,return_snac=True,return_basis=True)
        self.compute_fn=model_jit(self._raw)
        self.compute_batch_fn=model_jit(jax.vmap(self._raw))

    @property
    def electronic_dimension(self):
        return self.n_sf

    def electronic_overlap(self):
        if self.last_overlap is None:
            return np.eye(self.n_total_states)
        return _expand_sf_np(self.last_overlap,self.ns,self.nt)

    def graph_from_bohr(self,positions):
        positions=np.asarray(positions,dtype=float)
        if positions.shape!=(self.n_atoms,3) or not np.isfinite(positions).all():
            raise ValueError('Invalid SHARC positions')
        return self._create_graphs(positions if self.use_au else positions*units.Bohr)

    def calculate(self,positions):
        graph=self.graph_from_bohr(positions)
        raw=jax.device_get(self.compute_fn(graph))
        return self.from_raw(raw,np.asarray(graph.nodes.positions)[:self.n_atoms])

    def calculate_batch_raw(self,positions_bohr):
        graphs=[self.graph_from_bohr(x) for x in positions_bohr]
        stacked=jax.tree_util.tree_map(lambda *a:jnp.stack(a),*graphs)
        return jax.device_get(self.compute_batch_fn(stacked))

    def from_raw(self,raw,positions_model):
        e,f,soc,sn,u,_=[np.asarray(x) for x in raw]
        indices=spin_root_indices(self.ns,self.nt)
        i,j=self.nac_idx
        gaps=e[j]-e[i]
        if np.any(gaps<=self.safe_threshold):
            raise FloatingPointError('MCH gap too small for reliable NAC division')
        nfactor=1. if self.use_au else units.Bohr
        raw_pair=sn[:,:self.n_atoms]/gaps[:,None,None]
        pair_factors,_=self._align_dot_phase(u,raw_pair*nfactor)
        signs=self.last_phase_signs
        sp=signs[indices]
        soc=soc*sp[:,None]*sp[None,:]
        nac_sf=np.zeros((self.n_sf,self.n_sf,self.n_atoms,3))
        pair=raw_pair*pair_factors[:,None,None]
        nac_sf[i,j]=pair;nac_sf[j,i]=-pair
        efactor=1. if self.use_au else 1./units.Hartree
        gfactor=1. if self.use_au else units.Bohr/units.Hartree
        h=(np.diag((e+self.energy_offset)[indices])+soc)*efactor
        result=dict(h=h,grad=-f[indices,:self.n_atoms]*gfactor,
             nacdr=_expand_sf_np(nac_sf,self.ns,self.nt)*nfactor,
             overlap=self.electronic_overlap().astype(complex))
        result.update(
            energy=(e + self.energy_offset) * efactor,
            gradients=-f[:,:self.n_atoms] * gfactor,
            nacs=pair * nfactor,
            snacs=sn[:,:self.n_atoms] * pair_factors[:,None,None] * gfactor,
            soc=soc * efactor,
        )
        if any(not np.isfinite(x).all() for x in result.values()):
            raise FloatingPointError('Nonfinite MCH model output')
        return result

    def get_qm(self,prediction):
        return {key: prediction[key] for key in ("h", "grad", "nacdr", "overlap")}
