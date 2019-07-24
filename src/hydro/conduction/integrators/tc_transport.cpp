#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../parameter_input.hpp"
#include "../../../mesh/mesh.hpp"
#include "../tc.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../hydro.hpp"
#include "tc_integrators.hpp"
#include "../../../eos/eos.hpp"

void TCIntegrator::CalculateFluxes(MeshBlock *pmb,
                                   AthenaArray<Real> &w,
                                   AthenaArray<Real> &bcc,
                                   AthenaArray<Real> &u_tc,
                                   int reconstruct_order)
{
  auto ptc = pmy_tc;
  auto pco = pmb->pcoord;

  auto &x1flux = ptc->flux[X1DIR];
  auto &x2flux = ptc->flux[X2DIR];
  auto &x3flux = ptc->flux[X3DIR];

  AthenaArray<Real> flx, wl, wr, vdiff_l, vdiff_r, rhol, rhor, tl, tr, cwidth;
  flx.InitWithShallowCopy(flx_);
  vdiff_l.InitWithShallowCopy(vdiff_l_);
  vdiff_r.InitWithShallowCopy(vdiff_r_);
  wl.InitWithShallowCopy(wl_);
  wr.InitWithShallowCopy(wr_);
  rhol.InitWithShallowCopy(rho_l_);
  rhor.InitWithShallowCopy(rho_r_);
  tl.InitWithShallowCopy(tgas_l_);
  tr.InitWithShallowCopy(tgas_r_);
  cwidth.InitWithShallowCopy(cwidth_);

  int n1 = pmb->block_size.nx1 + 2*NGHOST;
  int n2 = pmb->block_size.nx2;
  if (n2 > 1) n2 += 2*NGHOST;
  int n3 = pmb->block_size.nx3;
  if (n3 > 1) n3 += 2*NGHOST;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  // First, calculate the diffusion velocity along three coordinate system
  // We first assume B is along x coordinate
  // Then rotate according to B direction to the actual acooridnate
  for (int k=0; k<n3; ++k) {
    for (int j=0; j<n2; ++j) {
      // diffusion velocity along the direction of sigma vector
      for (int i=0; i<n1; ++i) {
        Real kappa = ptc->rho(k,j,i)/ptc->tc_kappa(0,k,j,i);
        Real taux = ptc->vmax * kappa * pco->dx1f(i);
        taux = taux * taux/2.0;
        Real diffv = sqrt((1.0 - exp(-taux)) / taux);
        if (taux < 1.e-3) diffv = sqrt((1.0 - 0.5* taux));
        vdiff_(0,k,j,i) = ptc->vmax * diffv;
      }

      // y direction
      pco->CenterWidth2(k,j,0,n1-1,cwidth);
        // get the optical depth across the cell
      for (int i=0; i<n1; ++i) {
        Real kappa = ptc->rho(k,j,i)/ptc->tc_kappa(1,k,j,i);
        Real tauy = ptc->vmax * kappa * cwidth(i);
        tauy = tauy * tauy/2.0;
        Real diffv = sqrt((1.0 - exp(-tauy)) / tauy);
        if (tauy < 1.e-3) diffv = sqrt((1.0 - 0.5* tauy));
        vdiff_(1,k,j,i) = ptc->vmax * diffv;
      }// end i

      // z direction
      pco->CenterWidth3(k,j,0,n1-1,cwidth);
        // get the optical depth across the cell
      for (int i=0; i<n1; ++i) {
        Real kappa = ptc->rho(k,j,i)/ptc->tc_kappa(2,k,j,i);
        Real tauz = ptc->vmax * kappa * cwidth(i);
        tauz = tauz * tauz/2.0;
        Real diffv = sqrt((1.0 - exp(-tauz)) / tauz);
        if(tauz < 1.e-3) diffv = sqrt((1.0 - 0.5* tauz));
        vdiff_(2,k,j,i) = ptc->vmax * diffv;
      }

      //rotate the v_diff vector to the local coordinate
      if (MAGNETIC_FIELDS_ENABLED) {
        for (int i=0; i<n1; ++i) {
          Real sint_b = ptc->b_angle(0,k,j,i);
          Real cost_b = ptc->b_angle(1,k,j,i);
          Real sinp_b = ptc->b_angle(2,k,j,i);
          Real cosp_b = ptc->b_angle(3,k,j,i);
          InvRotateVec(sint_b,cost_b,sinp_b,cosp_b,
                       vdiff_(0,k,j,i),
                       vdiff_(1,k,j,i),
                       vdiff_(2,k,j,i));

          // take the absolute value
          vdiff_(0,k,j,i) = fabs(vdiff_(0,k,j,i));
          vdiff_(1,k,j,i) = fabs(vdiff_(1,k,j,i));
          vdiff_(2,k,j,i) = fabs(vdiff_(2,k,j,i));
        }
      }
    }
  }

  // i-direction
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      // Reconstruct density temperature and speed at left and right state
      if (reconstruct_order == 1) {
        DonorCellX1(k,j,is,ie+1,u_tc,ptc->rho,ptc->tgas,rhol,rhor,tl,tr,wl,wr);
      } else {
        PieceWiseLinear(k,j,is,ie+1,u_tc,ptc->rho,ptc->tgas,
                                rhol,rhor,tl,tr,wl,wr,0);
      }
      // get the optical depth across the cell
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        vdiff_l(i) = -vdiff_(0,k,j,i-1);
        vdiff_r(i) = vdiff_(0,k,j,i);
      }

      // calculate the flux
      TCFlux(1, is, ie+1, tl, tr, rhol,rhor, wl, wr, vdiff_l, vdiff_r, flx);
      // store the flux
      for (int n=0; n<NTC; ++n) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          x1flux(n,k,j,i) = flx(n,i);
        }
      }
    }
  }

  // j-direction
  if (n2 > 1) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        if (reconstruct_order == 1) {
          DonorCellX2(k,j,is,ie,u_tc,ptc->rho,ptc->tgas,rhol,rhor,tl,tr,wl,wr);
        } else {
          PieceWiseLinear(k,j,is,ie,u_tc,ptc->rho,ptc->tgas,
                          rhol,rhor,tl,tr,wl,wr,1);
        }

        // get the optical depth across the cell
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          vdiff_l(i) = -vdiff_(1,k,j-1,i);
          vdiff_r(i) = vdiff_(1,k,j,i);
        }

        // calculate the flux
        TCFlux(2, is, ie, tl, tr, rhol, rhor, wl,wr, vdiff_l, vdiff_r, flx);

        // store the flux
        for (int n=0; n<NTC; ++n) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            x2flux(n,k,j,i) = flx(n,i);
          }
        }
      }
    }
  }// finish j direction

  //  k-direction
  if (n3 > 1) {
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        if (reconstruct_order == 1) {
          DonorCellX3(k,j,is,ie,u_tc,ptc->rho,ptc->tgas,rhol,rhor,tl,tr,wl,wr);
        } else {
          PieceWiseLinear(k,j,is,ie,u_tc,ptc->rho,ptc->tgas,
                          rhol,rhor,tl,tr,wl,wr,2);
        }

        // get the optical depth across the cell
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          vdiff_l(i) = -vdiff_(2,k-1,j,i);
          vdiff_r(i) = vdiff_(2,k,j,i);
        }

        // calculate the flux
        TCFlux(3, is, ie, tl, tr, rhol, rhor, wl, wr, vdiff_l, vdiff_r, flx);

        // store the flux
        for (int n=0; n<NTC; ++n) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            x3flux(n,k,j,i) = flx(n,i);
          }
        }
      }
    }
  }// finish k direction
}

//  Adds flux divergence to weighted average of conservative variables from
//  previous step(s) of time integrator algorithm
void TCIntegrator::AddFluxDivergenceToAverage(MeshBlock *pmb,
                                              AthenaArray<Real> &u_tc,
                                              Real const weight)
{
  auto *ptc = pmy_tc;
  auto *pco = pmb->pcoord;
  Real dt = pmb->pmy_mesh->dt;

  AthenaArray<Real> &x1flux = ptc->flux[X1DIR];
  AthenaArray<Real> &x2flux = ptc->flux[X2DIR];
  AthenaArray<Real> &x3flux = ptc->flux[X3DIR];

  AthenaArray<Real> x1area, x2area, x2area_p1, x3area, x3area_p1, vol, dflx;
  x1area.InitWithShallowCopy(x1face_area_);
  x2area.InitWithShallowCopy(x2face_area_);
  x2area_p1.InitWithShallowCopy(x2face_area_p1_);
  x3area.InitWithShallowCopy(x3face_area_);
  x3area_p1.InitWithShallowCopy(x3face_area_p1_);
  vol.InitWithShallowCopy(cell_volume_);
  dflx.InitWithShallowCopy(flx_);

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {

      // calculate x1-flux divergence
      pmb->pcoord->Face1Area(k,j,is,ie+1,x1area);
      for (int n=0; n<NTC; ++n) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          dflx(n,i) = x1area(i+1)*x1flux(n,k,j,i+1) -
                      x1area(i)*x1flux(n,k,j,i);
        }
      }

     // calculate x2-flux divergence
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face2Area(k,j  ,is,ie,x2area   );
        pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);
        for (int n=0; n<NTC; ++n) {
#pragma omp simd
          for (int i=is; i<=ie; ++i ){
            dflx(n,i) += x2area_p1(i)*x2flux(n,k,j+1,i) -
                         x2area(i)*x2flux(n,k,j,i);
          }
        }
      }

      // calculate x3-flux divergence
      if (pmb->block_size.nx3 > 1) {
        pmb->pcoord->Face3Area(k  ,j,is,ie,x3area   );
        pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
        for (int n=0; n<NTC; ++n) {
#pragma omp simdf
          for (int i=is; i<=ie; ++i) {
            dflx(n,i) += x3area_p1(i)*x3flux(n,k+1,j,i) -
                         x3area(i)*x3flux(n,k,j,i);
          }
        }
      }

      // update variable with flux divergence
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
      for (int n=0; n<NTC; ++n) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          u_tc(n,k,j,i) -= weight*dt*ptc->rho(k,j,i)*dflx(n,i)/vol(i);
        }
      }

      // get the energy source term
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        tc_esource_(k,j,i) = -weight*dt*dflx(0,i)/vol(i);
      }

    }// end j
  }// end k
}

//  Compute weighted average of cell-averaged U in time integrator step
//  i.e. Sets U = a*U + b*U1 + c*U2
void TCIntegrator::WeightedAveU(MeshBlock* pmb,
                                AthenaArray<Real> &u,
                                AthenaArray<Real> &u1,
                                AthenaArray<Real> &u2,
                                const Real weights[3])
{
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real a = weights[0];
  Real b = weights[1];
  Real c = weights[2];

  // if c=0, c=b=0, or c=b=a=0 (in that order) to avoid extra FMA operations
  if (c != 0.0) {
    for (int n=0;  n<NTC; ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
    for (int i=is; i<=ie; ++i) {
      u(n,k,j,i) = a*u(n,k,j,i) + b*u1(n,k,j,i) + c*u2(n,k,j,i);
    }}}}
  } else if (b != 0.0) { // do not dereference u_in2
    for (int n=0;  n<NTC; ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
    for (int i=is; i<=ie; ++i) {
      u(n,k,j,i) = a*u(n,k,j,i) + b*u1(n,k,j,i);
    }}}}
  } else if (a != 0.0) { // do not dereference u_in1
    for (int n=0;  n<NTC; ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
    for (int i=is; i<=ie; ++i) {
      u(n,k,j,i) = a*u(n,k,j,i);
    }}}}
  } else { // directly initialize u_out to 0
    for (int n=0;  n<NTC; ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
    for (int i=is; i<=ie; ++i) {
      u(n,k,j,i) = 0.0;
    }}}}
  }
}
