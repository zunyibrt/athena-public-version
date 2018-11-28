// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../cr.hpp"
#include "../../hydro/hydro.hpp"
#include "../../field/field.hpp"
#include "../../eos/eos.hpp"
#include "cr_integrators.hpp"

// Function for adding the source terms implicitly
void CRIntegrator::AddSourceTerms(MeshBlock *pmb, const Real dt, AthenaArray<Real> &u,
                                  AthenaArray<Real> &w, AthenaArray<Real> &u_cr) {
  CosmicRay *pcr=pmb->pcr;
  
  Real vmax = pcr->vmax;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
 
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
	Real v1 = w(IVX,k,j,i);
        Real v2 = w(IVY,k,j,i);
        Real v3 = w(IVZ,k,j,i);

        Real fxx = pcr->prtensor_cr(PC11,k,j,i);
        Real fyy = pcr->prtensor_cr(PC22,k,j,i);
        Real fzz = pcr->prtensor_cr(PC33,k,j,i);
        Real fxy = pcr->prtensor_cr(PC12,k,j,i);
        Real fxz = pcr->prtensor_cr(PC13,k,j,i);
        Real fyz = pcr->prtensor_cr(PC23,k,j,i);

        Real ec  = u_cr(CRE ,k,j,i);
        Real fc1 = u_cr(CRF1,k,j,i);
        Real fc2 = u_cr(CRF2,k,j,i);
        Real fc3 = u_cr(CRF3,k,j,i);
	Real fr1=fc1; Real fr2=fc2; Real fr3=fc3;

        Real sint_b = pcr->b_angle(0,k,j,i);
        Real cost_b = pcr->b_angle(1,k,j,i);
        Real sinp_b = pcr->b_angle(2,k,j,i);
        Real cosp_b = pcr->b_angle(3,k,j,i);

        // Rotate the vectors to orientate to the B direction
	// Pressure is assumed to be Isotropic
        RotateVec(sint_b,cost_b,sinp_b,cosp_b,v1,v2,v3);
        RotateVec(sint_b,cost_b,sinp_b,cosp_b,fr1,fr2,fr3);

	// Compute Sigmas
        Real sigma_x = 1.0/(1.0/pcr->sigma_diff(0,k,j,i)
	                   +1.0/pcr->sigma_adv( 0,k,j,i));

        Real sigma_y = 1.0/(1.0/pcr->sigma_diff(1,k,j,i) 
                           +1.0/pcr->sigma_adv( 1,k,j,i));

        Real sigma_z = 1.0/(1.0/pcr->sigma_diff(2,k,j,i) 
                           +1.0/pcr->sigma_adv( 2,k,j,i));

        // Compute new Fluxes in rotated frame
	Real dtsigma1 = dt*sigma_x*vmax;
        Real dtsigma2 = dt*sigma_y*vmax;
        Real dtsigma3 = dt*sigma_z*vmax;

        Real rhs1 = (v1*(1.0 + fxx) + v2*fxy + v3*fxz) * ec/vmax;
        Real rhs2 = (v2*(1.0 + fyy) + v1*fxy + v3*fyz) * ec/vmax;
        Real rhs3 = (v3*(1.0 + fzz) + v1*fxz + v2*fyz) * ec/vmax;

        Real newfr1 = (fr1 - rhs1)/(1.0 + dtsigma1) + rhs1;
        Real newfr2 = (fr2 - rhs2)/(1.0 + dtsigma2) + rhs2;
        Real newfr3 = (fr3 - rhs3)/(1.0 + dtsigma3) + rhs3;         

        // Now apply the invert rotation
        InvRotateVec(sint_b,cost_b,sinp_b,cosp_b,newfr1,newfr2,newfr3);

        // Update Cosmic Ray quantities
	u_cr(CRE ,k,j,i) = ec + dt*ec_source_(k,j,i);
        u_cr(CRF1,k,j,i) = newfr1;
        u_cr(CRF2,k,j,i) = newfr2;
        u_cr(CRF3,k,j,i) = newfr3;

        // Add source term to gas
        u(IEN,k,j,i) -= dt*ec_source_(k,j,i);
        u(IM1,k,j,i) += (-(newfr1 - fc1) / vmax);
        u(IM2,k,j,i) += (-(newfr2 - fc2) / vmax);
        u(IM3,k,j,i) += (-(newfr3 - fc3) / vmax);

      }// end i
    }// end j
  }// end k
}
