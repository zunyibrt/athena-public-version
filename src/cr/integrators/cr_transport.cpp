#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../cr.hpp"
#include "../../coordinates/coordinates.hpp"
#include "cr_integrators.hpp"

void CRIntegrator::CalculateFluxes(MeshBlock *pmb,
                                   AthenaArray<Real> &w,
		                               AthenaArray<Real> &bcc,
                                   AthenaArray<Real> &u_cr,
				                           int reconstruct_order)
{
  auto pcr = pmy_cr;
  auto &x1flux = pcr->flux[X1DIR];
  auto &x2flux = pcr->flux[X2DIR];
  auto &x3flux = pcr->flux[X3DIR];

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  int n1 = pmb->block_size.nx1 + 2*NGHOST;
  int n2 = pmb->block_size.nx2;
  if (n2 > 1) n2 += 2*NGHOST;
  int n3 = pmb->block_size.nx3;
  if (n3 > 1) n3 += 2*NGHOST;

  AthenaArray<Real> flx, vel_l, vel_r, wl, wr, vdiff_l, vdiff_r, crptl, crptr;
  AthenaArray<Real> cwidth;
  flx.InitWithShallowCopy(flx_);
  vel_l.InitWithShallowCopy(vel_l_);
  vel_r.InitWithShallowCopy(vel_r_);
  vdiff_l.InitWithShallowCopy(vdiff_l_);
  vdiff_r.InitWithShallowCopy(vdiff_r_);
  crptl.InitWithShallowCopy(crptl_);
  crptr.InitWithShallowCopy(crptr_);
  wl.InitWithShallowCopy(wl_);
  wr.InitWithShallowCopy(wr_);
  cwidth.InitWithShallowCopy(cwidth_);

  // The area functions needed to calculate Grad Pc
  AthenaArray<Real> x1area, x2area, x2area_p1, x3area, x3area_p1, vol;
  x1area.InitWithShallowCopy(x1face_area_);
  x2area.InitWithShallowCopy(x2face_area_);
  x2area_p1.InitWithShallowCopy(x2face_area_p1_);
  x3area.InitWithShallowCopy(x3face_area_);
  x3area_p1.InitWithShallowCopy(x3face_area_p1_);
  vol.InitWithShallowCopy(cell_volume_);

//--------------------------------------------------------------------------------------
  // First, calculate the diffusion velocity along three coordinate system
  for (int k=0; k<n3; ++k){
    for(int j=0; j<n2; ++j){
      // diffusion velocity along the direction of sigma vector
      // We first assume B is along x coordinate
      // Then rotate according to B direction to the actual coordinate
      for (int i=0; i<n1; ++i) {
        Real crpt_xx = pcr->prtensor_cr(PC11,k,j,i);
        Real totsigma = 1.0/(1.0/pcr->sigma_diff(0,k,j,i)
                           + 1.0/pcr->sigma_adv(0,k,j,i));
        Real taux = totsigma * pmb->pcoord->dx1f(i);
        taux = taux * taux/(2.0 * crpt_xx);
        Real diffv = sqrt((1.0 - exp(-taux)) / taux);

	      if(taux < 1.e-3) diffv = sqrt((1.0 - 0.5* taux));
        pcr->v_diff(0,k,j,i) = pcr->vmax * sqrt(crpt_xx) * diffv;
      }

      // y direction
      pmb->pcoord->CenterWidth2(k,j,0,n1-1,cwidth);
      for (int i=0; i<n1; ++i) {
        Real crpt_yy = pcr->prtensor_cr(PC22,k,j,i);
        Real totsigma = 1.0/(1.0/pcr->sigma_diff(1,k,j,i)
                           + 1.0/pcr->sigma_adv(1,k,j,i));
        Real tauy = totsigma * cwidth(i);
        tauy = tauy * tauy/(2.0 * crpt_yy);
        Real diffv = sqrt((1.0 - exp(-tauy)) / tauy);

        if(tauy < 1.e-3) diffv = sqrt((1.0 - 0.5* tauy));
        pcr->v_diff(1,k,j,i) = pcr->vmax * sqrt(crpt_yy) * diffv;
      }

      // z direction
      pmb->pcoord->CenterWidth3(k,j,0,n1-1,cwidth);
      for (int i=0; i<n1; ++i) {
        Real crpt_zz = pcr->prtensor_cr(PC33,k,j,i);
        Real totsigma = 1.0/(1.0/pcr->sigma_diff(2,k,j,i)
                           + 1.0/pcr->sigma_adv(2,k,j,i));
        Real tauz = totsigma * cwidth(i);
        tauz = tauz * tauz/(2.0 * crpt_zz);
        Real diffv = sqrt((1.0 - exp(-tauz)) / tauz);

        if(tauz < 1.e-3) diffv = sqrt((1.0 - 0.5* tauz));
        pcr->v_diff(2,k,j,i) = pcr->vmax * sqrt(crpt_zz) * diffv;
      }

      // rotate the v_diff vector to the local coordinate
      for (int i=0; i<n1; ++i) {
        Real sint_b = pcr->b_angle(0,k,j,i);
        Real cost_b = pcr->b_angle(1,k,j,i);
        Real sinp_b = pcr->b_angle(2,k,j,i);
        Real cosp_b = pcr->b_angle(3,k,j,i);
        InvRotateVec(sint_b,cost_b,sinp_b,cosp_b,
                     pcr->v_diff(0,k,j,i),
                     pcr->v_diff(1,k,j,i),
                     pcr->v_diff(2,k,j,i));
      }

	      // take the absolute value
        // Also add the Alfven velocity for the streaming flux
#pragma omp simd
      for (int i=0; i<n1; ++i) {
        pcr->v_diff(0,k,j,i) = fabs(pcr->v_diff(0,k,j,i));
        pcr->v_diff(1,k,j,i) = fabs(pcr->v_diff(1,k,j,i));
        pcr->v_diff(2,k,j,i) = fabs(pcr->v_diff(2,k,j,i));
      }

      // need to add additional sound speed for stability
      for (int i=0; i<n1; ++i) {
         Real cr_sound_x = sqrt((4.0/3.0) * pcr->prtensor_cr(PC11,k,j,i)
                           * u_cr(k,j,i) / w(IDN,k,j,i));
        pcr->v_diff(0,k,j,i) += cr_sound_x;

         Real cr_sound_y = sqrt((4.0/3.0) * pcr->prtensor_cr(PC22,k,j,i) *
                           u_cr(k,j,i) / w(IDN,k,j,i));
         pcr->v_diff(1,k,j,i) += cr_sound_y;

         Real cr_sound_z = sqrt((4.0/3.0) * pcr->prtensor_cr(PC33,k,j,i) *
                           u_cr(k,j,i)/w(IDN,k,j,i));
         pcr->v_diff(2,k,j,i) += cr_sound_z;
      }
    }
  }

  //--------------------------------------------------------------------------------------
  // i-direction
  for (int k=ks; k<=ke; ++k){
    for (int j=js; j<=je; ++j){
      // Reconstruction of Ec, Fc, and vel
      if(reconstruct_order == 1){
        DonorCellX1(k,j,is,ie+1,u_cr,w,pcr->prtensor_cr,
                    wl,wr,vel_l,vel_r,crptl,crptr);
      } else {
        PieceWiseLinear(k,j,is,ie+1,u_cr,w,pcr->prtensor_cr,
                        wl,wr,vel_l,vel_r,crptl,crptr,X1DIR);
      }

#pragma omp simd
      for(int i=is; i<=ie+1; ++i){
        vdiff_l(i) = pcr->v_diff(0,k,j,i-1);
        vdiff_r(i) = pcr->v_diff(0,k,j,i);
      }

      // Riemann Solver
      CRFlux(CRF1, k, j, is, ie+1, wl, wr, vel_l, vel_r,
             crptl, crptr, vdiff_l, vdiff_r, flx);

      // Store computed flux
      for(int n=0; n<NCR; ++n){
#pragma omp simd
        for(int i=is; i<=ie+1; ++i){
          x1flux(n,k,j,i) = flx(n,i);
        }
      }
    }
  }//finish i-direction

  // j-direction
  if (pmb->block_size.nx2 > 1) {
    for (int k=ks; k<=ke; ++k){
      for (int j=js; j<=je+1; ++j){
        // Reconstruction of Ec, Fc, and vel
        if(reconstruct_order == 1){
          DonorCellX2(k,j,is,ie,u_cr,w,pcr->prtensor_cr,
                      wl,wr,vel_l,vel_r,crptl,crptr);
        } else {
          PieceWiseLinear(k,j,is,ie,u_cr,w,pcr->prtensor_cr,
                          wl,wr,vel_l,vel_r,crptl,crptr,X2DIR);
        }

#pragma omp simd
        for(int i=is; i<=ie; ++i){
          vdiff_l(i) = pcr->v_diff(1,k,j-1,i);
          vdiff_r(i) = pcr->v_diff(1,k,j,i);
        }

        // Riemann Solver
        CRFlux(CRF2, k, j, is, ie, wl, wr, vel_l, vel_r,
               crptl, crptr, vdiff_l, vdiff_r, flx);

        // Store computed flux
        for(int n=0; n<NCR; ++n){
#pragma omp simd
          for(int i=is; i<=ie; ++i){
            x2flux(n,k,j,i) = flx(n,i);
          }
        }
      }
    }
  }// finish j direction

  // k-direction
  if (pmb->block_size.nx3 > 1) {
    for (int k=ks; k<=ke+1; ++k){
      for (int j=js; j<=je; ++j){
        // Reconstruction of Ec, Fc, and vel
        if(reconstruct_order == 1){
          DonorCellX3(k,j,is,ie,u_cr,w,pcr->prtensor_cr,
                      wl,wr,vel_l,vel_r,crptl,crptr);
        } else {
          PieceWiseLinear(k,j,is,ie,u_cr,w,pcr->prtensor_cr,
                          wl,wr,vel_l,vel_r,crptl,crptr,X3DIR);
        }

#pragma omp simd
        for(int i=is; i<=ie; ++i){
          vdiff_l(i) = pcr->v_diff(2,k-1,j,i);
          vdiff_r(i) = pcr->v_diff(2,k,j,i);
        }
        // Riemann Solver
        CRFlux(CRF3, k, j, is, ie, wl, wr, vel_l, vel_r,
               crptl, crptr, vdiff_l, vdiff_r, flx);

        // Store computed flux
        for(int n=0; n<NCR; ++n){
#pragma omp simd
          for(int i=is; i<=ie; ++i){
            x3flux(n,k,j,i) = flx(n,i);
          }
        }
      }
    }
  }// finish k direction

  // Now calculate Grad Pc and the associated heating term
  //--------------------------------------------------------------------------------//
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {

        pmb->pcoord->CellVolume(k,j,is,ie,vol);

        // x1 direction
        pmb->pcoord->Face1Area(k,j,is,ie+1,x1area);
        for (int n=0; n<3; ++n) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            grad_pc_(n,k,j,i) = (x1area(i+1)*x1flux(CRF1+n,k,j,i+1)
                                 - x1area(i)*x1flux(CRF1+n,k,j,i))/vol(i);
          }
        }

        // x2 direction
        if (pmb->block_size.nx2 > 1) {
          pmb->pcoord->Face2Area(k,j,is,ie,x2area);
          pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);
          for (int n=0; n<3; ++n) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              grad_pc_(n,k,j,i) += (x2area_p1(i)*x2flux(CRF1+n,k,j+1,i)
                                    - x2area(i)*x2flux(CRF1+n,k,j,i))/vol(i);
            }
          }
        }

        // x3 direction
        if (pmb->block_size.nx3 > 1) {
          pmb->pcoord->Face3Area(k,j,is,ie,x3area);
          pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
          for (int n=0; n<3; ++n) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              grad_pc_(n,k,j,i) += (x3area_p1(i)*x3flux(CRF1+n,k+1,j,i)
                                    - x3area(i)*x3flux(CRF1+n,k,j,i))/vol(i);
            }
          }
        }


        for(int n=0; n<3; ++n){
#pragma omp simd
          for(int i=is; i<=ie; ++i){
            grad_pc_(n,k,j,i) /= pcr->vmax;
          }
        }

        // calculate source term and add it explicitly for Ec
        for (int i=is; i<=ie; ++i) {
          // Compute Streaming velocityies
          Real sqrt_rho = sqrt(w(IDN,k,j,i));
          Real va1 = bcc(IB1,k,j,i)/sqrt_rho;
          Real va2 = bcc(IB2,k,j,i)/sqrt_rho;
          Real va3 = bcc(IB3,k,j,i)/sqrt_rho;

          Real b_grad_pc = bcc(IB1,k,j,i) * grad_pc_(0,k,j,i) +
                           bcc(IB2,k,j,i) * grad_pc_(1,k,j,i) +
                           bcc(IB3,k,j,i) * grad_pc_(2,k,j,i);
          Real dpc_sign = 0.0;
          if (fabs(b_grad_pc) > TINY_NUMBER) dpc_sign = SIGN(b_grad_pc);

          pcr->v_adv(0,k,j,i) = -va1 * dpc_sign;
          pcr->v_adv(1,k,j,i) = -va2 * dpc_sign;
          pcr->v_adv(2,k,j,i) = -va3 * dpc_sign;

          // Compute sigma advection wrt B Field
          Real btot = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i) +
                           bcc(IB3,k,j,i)*bcc(IB3,k,j,i));
          Real va = btot/sqrt_rho;
          if (va > TINY_NUMBER) {
            pcr->sigma_adv(0,k,j,i) = pcr->vmax * fabs(b_grad_pc) /
                                      (btot * va * u_cr(CRE,k,j,i) *
                                      (1.0 + pcr->prtensor_cr(PC11,k,j,i)));
          }

          // Compute total sigma
          Real sigma_x = 1.0/(1.0/pcr->sigma_diff(0,k,j,i)
                            +1.0/pcr->sigma_adv(0,k,j,i));

          Real v1 = w(IVX,k,j,i);
          Real v2 = w(IVY,k,j,i);
          Real v3 = w(IVZ,k,j,i);

          Real vtotx = v1 + pcr->v_adv(0,k,j,i);
          Real vtoty = v2 + pcr->v_adv(1,k,j,i);
          Real vtotz = v3 + pcr->v_adv(2,k,j,i);

          Real dpcdx = grad_pc_(0,k,j,i);
          Real dpcdy = grad_pc_(1,k,j,i);
          Real dpcdz = grad_pc_(2,k,j,i);

          // explicit method needs to use CR flux from previous step
          Real fr1 = u_cr(CRF1,k,j,i);
          Real fr2 = u_cr(CRF2,k,j,i);
          Real fr3 = u_cr(CRF3,k,j,i);

          // Rotate to frame of bfield
          Real sint_b = pcr->b_angle(0,k,j,i);
          Real cost_b = pcr->b_angle(1,k,j,i);
          Real sinp_b = pcr->b_angle(2,k,j,i);
          Real cosp_b = pcr->b_angle(3,k,j,i);

          RotateVec(sint_b,cost_b,sinp_b,cosp_b,vtotx,vtoty,vtotz);
          RotateVec(sint_b,cost_b,sinp_b,cosp_b,dpcdx,dpcdy,dpcdz);
          RotateVec(sint_b,cost_b,sinp_b,cosp_b,v1,v2,v3);
          RotateVec(sint_b,cost_b,sinp_b,cosp_b,fr1,fr2,fr3);

          // only calculate v_dot_gradpc perpendicular to B
          // perpendicular direction only has flow velocity, no streaming
          Real v_dot_gradpc = v2 * dpcdy + v3 * dpcdz;

          Real fxx = pcr->prtensor_cr(PC11,k,j,i);
          Real fxy = pcr->prtensor_cr(PC12,k,j,i);
          Real fxz = pcr->prtensor_cr(PC13,k,j,i);

          Real fr_cm1 = fr1 - (v1 * (1.0 + fxx) + v2 * fxy + v3 * fxz)
                        * u_cr(CRE,k,j,i) / pcr->vmax;

          ec_source_(k,j,i) = (v_dot_gradpc - vtotx * sigma_x * fr_cm1);
        }// end i
      }// end k
    }// end j
}

//----------------------------------------------------------------------------------------
//  Compute weighted average of cell-averaged U in time integrator step
void CRIntegrator::WeightedAveU(MeshBlock* pmb,
                                AthenaArray<Real> &u_out,
                                AthenaArray<Real> &u_in1,
                                AthenaArray<Real> &u_in2,
                                const Real weights[3])
{
  // consider every possible simplified form of weighted sum operator:
  // U = a*U + b*U1 + c*U2
  // if c=0, c=b=0, or c=b=a=0 (in that order) to avoid extra FMA operations
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  if (weights[2] != 0.0) {
    for (int n=0; n<NCR; ++n) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            u_out(n,k,j,i) = weights[0]*u_out(n,k,j,i) +
                             weights[1]*u_in1(n,k,j,i) +
                             weights[2]*u_in2(n,k,j,i);
    } } } }
  } else { // do not dereference u_in2
    if (weights[1] != 0.0) {
      for (int n=0; n<NCR; ++n) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              u_out(n,k,j,i) = weights[0]*u_out(n,k,j,i) +
                               weights[1]*u_in1(n,k,j,i);
      } } } }
    } else { // do not dereference u_in1
      if (weights[0] != 0.0) {
        for (int n=0; n<NCR; ++n) {
          for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                u_out(n,k,j,i) = weights[0]*u_out(n,k,j,i);
        } } } }
      } else { // directly initialize u_out to 0
        for (int n=0; n<NCR; ++n) {
          for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                u_out(n,k,j,i) = 0.0;
        } } } }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//  Adds flux divergence to weighted average of conservative variables from
//  previous step(s) of time integrator algorithm
void CRIntegrator::AddFluxDivergenceToAverage(MeshBlock *pmb,
                                              AthenaArray<Real> &u_cr,
                                              Real const weight)
{
  auto &x1flux = pmb->pcr->flux[X1DIR];
  auto &x2flux = pmb->pcr->flux[X2DIR];
  auto &x3flux = pmb->pcr->flux[X3DIR];

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real dt = pmb->pmy_mesh->dt;

  AthenaArray<Real> x1area, x2area, x2area_p1, x3area, x3area_p1, vol, dflx;
  x1area.InitWithShallowCopy(x1face_area_);
  x2area.InitWithShallowCopy(x2face_area_);
  x2area_p1.InitWithShallowCopy(x2face_area_p1_);
  x3area.InitWithShallowCopy(x3face_area_);
  x3area_p1.InitWithShallowCopy(x3face_area_p1_);
  vol.InitWithShallowCopy(cell_volume_);
  dflx.InitWithShallowCopy(flx_);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {

      // calculate x1-flux divergence
      pmb->pcoord->Face1Area(k,j,is,ie+1,x1area);
      for (int n=0; n<NCR; ++n) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          dflx(n,i) = x1area(i+1)*x1flux(n,k,j,i+1) -
                      x1area(i)*x1flux(n,k,j,i);
        }
      }

      // calculate x2-flux divergence
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face2Area(k,j,is,ie,x2area);
        pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);
        for (int n=0; n<NCR; ++n) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            dflx(n,i) += x2area_p1(i)*x2flux(n,k,j+1,i) -
                         x2area(i)*x2flux(n,k,j,i);
          }
        }
      }

      // calculate x3-flux divergence
      if (pmb->block_size.nx3 > 1) {
        pmb->pcoord->Face3Area(k,j,is,ie,x3area);
        pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
        for (int n=0; n<NCR; ++n) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            dflx(n,i) += x3area_p1(i)*x3flux(n,k+1,j,i) -
                         x3area(i)*x3flux(n,k,j,i);
          }
        }
      }

      // update variables with flux divergence
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
      for(int n=0; n<NCR; ++n){
#pragma omp simd
        for(int i=is; i<=ie; ++i){
          u_cr(n,k,j,i) -= weight*dt*dflx(n,i)/vol(i);
        }
      }

      // Check that cosmic ray energy density is always positive
#pragma omp simd
      for(int i=is; i<=ie; ++i){
        u_cr(CRE,k,j,i) = std::max(u_cr(CRE,k,j,i), TINY_NUMBER);
      }

    }// end j
  }// end k

  // Note: Coordinate source terms not implemented, will not work for
  //       non-cartesian geometries
}
