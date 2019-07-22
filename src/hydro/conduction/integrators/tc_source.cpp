// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../parameter_input.hpp"
#include "../../../mesh/mesh.hpp"
#include "../tc.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../../field/field.hpp"
#include "tc_integrators.hpp"

// Add Source Terms implicitly
void TCIntegrator::AddSourceTerms(MeshBlock *pmb,
                                  Real const dt,
                                  AthenaArray<Real> &u,
                                  AthenaArray<Real> &u_tc)
{
  auto ptc = pmy_tc;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {

         Real f1 = u_tc(1,k,j,0);
         Real f2 = u_tc(2,k,j,0);
         Real f3 = u_tc(3,k,j,0);

         // Compute rotation matrix components
         Real sint_b = ptc->b_angle(0,k,j,i);
         Real cost_b = ptc->b_angle(1,k,j,i);
         Real sinp_b = ptc->b_angle(2,k,j,i);
         Real cosp_b = ptc->b_angle(3,k,j,i);

         // Rotate flux vectors to orientate to the B direction
         if (MAGNETIC_FIELDS_ENABLED) {
           RotateVec(sint_b,cost_b,sinp_b,cosp_b,f1,f2,f3);
         }

         // Now update the momentum equation to compute new fluxes
         // partial F/partial t = - rho * (V_m^2/kappa) * F
         // The solution is F_new = F_old/(1 + dt*rho*V_m^2/kappa)
         Real rho     = ptc->rho(k,j,i);
         Real vlim    = ptc->vmax;
         Real kappa_x = ptc->tc_kappa(0,k,j,i);
         Real kappa_y = ptc->tc_kappa(1,k,j,i);
         Real kappa_z = ptc->tc_kappa(2,k,j,i);

         f1 /= (1.0 + dt * rho * vlim * vlim/kappa_x);
         f2 /= (1.0 + dt * rho * vlim * vlim/kappa_y);
         f3 /= (1.0 + dt * rho * vlim * vlim/kappa_z);

         // Now apply the inverse rotation
         if (MAGNETIC_FIELDS_ENABLED) {
           InvRotateVec(sint_b,cost_b,sinp_b,cosp_b,f1,f2,f3);
         }

         // Update fluxes
         u_tc(1,k,j,i) = f1;
         u_tc(2,k,j,i) = f2;
         u_tc(3,k,j,i) = f3;

         // Update energy source term (calculated in transport step)
         u(IEN,k,j,i) += tc_esource_(k,j,i);

      }// end i
    }// end j
  }// end k
}
