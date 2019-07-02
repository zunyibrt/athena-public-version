// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../mesh/mesh.hpp"
#include "../cr.hpp"
#include "cr_integrators.hpp"

// Add Source Terms implicitly (Section 3.2.2, Jiang & Oh)
void CRIntegrator::AddSourceTerms(MeshBlock *pmb, Real const dt,
                                  AthenaArray<Real> &u,
                                  AthenaArray<Real> &w,
                                  AthenaArray<Real> &u_cr)
{
  CosmicRay *pcr = pmb->pcr;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {

       	Real v1 = w(IVX,k,j,i);
        Real v2 = w(IVY,k,j,i);
        Real v3 = w(IVZ,k,j,i);

        Real ec  = u_cr(CRE ,k,j,i);
        Real fc1 = u_cr(CRF1,k,j,i);
        Real fc2 = u_cr(CRF2,k,j,i);
        Real fc3 = u_cr(CRF3,k,j,i);

        Real fr1 = fc1;
        Real fr2 = fc2;
        Real fr3 = fc3;

        // Compute rotation matrix components
        Real sint_b = pcr->b_angle(0,k,j,i);
        Real cost_b = pcr->b_angle(1,k,j,i);
        Real sinp_b = pcr->b_angle(2,k,j,i);
        Real cosp_b = pcr->b_angle(3,k,j,i);

        // Rotate velocity and flux vectors to orientate to the B direction
        // Note: Rotate pressure tensor if it is not isotropic
        RotateVec(sint_b,cost_b,sinp_b,cosp_b,v1,v2,v3);
        RotateVec(sint_b,cost_b,sinp_b,cosp_b,fr1,fr2,fr3);

      	// Add Flux source term implicitly in rotated frame
        Real fxx = pcr->prtensor_cr(PC11,k,j,i);
        Real fyy = pcr->prtensor_cr(PC22,k,j,i);
        Real fzz = pcr->prtensor_cr(PC33,k,j,i);
        Real fxy = pcr->prtensor_cr(PC12,k,j,i);
        Real fxz = pcr->prtensor_cr(PC13,k,j,i);
        Real fyz = pcr->prtensor_cr(PC23,k,j,i);

        Real rhs1 = (v1*(1.0 + fxx) + v2*fxy + v3*fxz) * ec/pcr->vmax;
        Real rhs2 = (v2*(1.0 + fyy) + v1*fxy + v3*fyz) * ec/pcr->vmax;
        Real rhs3 = (v3*(1.0 + fzz) + v1*fxz + v2*fyz) * ec/pcr->vmax;

        Real sigma_x = 1.0/(1.0/pcr->sigma_diff(0,k,j,i)
	                         +1.0/pcr->sigma_adv(0,k,j,i));
        Real sigma_y = pcr->sigma_diff(1,k,j,i);
        Real sigma_z = pcr->sigma_diff(2,k,j,i);

        Real dtsigma1 = dt*sigma_x*pcr->vmax;
        Real dtsigma2 = dt*sigma_y*pcr->vmax;
        Real dtsigma3 = dt*sigma_z*pcr->vmax;

        fr1 = (fr1 - rhs1)/(1.0 + dtsigma1) + rhs1;
        fr2 = (fr2 - rhs2)/(1.0 + dtsigma2) + rhs2;
        fr3 = (fr3 - rhs3)/(1.0 + dtsigma3) + rhs3;

        // Now apply the invert rotation to get new fluxes in lab frame
        InvRotateVec(sint_b,cost_b,sinp_b,cosp_b,fr1,fr2,fr3);

        // Update Cosmic Ray quantities
        u_cr(CRE,k,j,i) += dt*ec_source_(k,j,i);
        u_cr(CRF1,k,j,i) = fr1;
        u_cr(CRF2,k,j,i) = fr2;
        u_cr(CRF3,k,j,i) = fr3;

        // Add source term to gas
        u(IEN,k,j,i) -= dt*ec_source_(k,j,i);
        u(IM1,k,j,i) -= (fr1-fc1)/pcr->vmax;
        u(IM2,k,j,i) -= (fr2-fc2)/pcr->vmax;
        u(IM3,k,j,i) -= (fr3-fc3)/pcr->vmax;

        // Ensure that Energy values are positive
        u_cr(CRE,k,j,i) = std::max(u_cr(CRE,k,j,i), TINY_NUMBER);
        u(IEN,k,j,i) = std::max(u(IEN,k,j,i), TINY_NUMBER);

      }// end i
    }// end j
  }// end k
}
