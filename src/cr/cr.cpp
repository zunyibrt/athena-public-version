// C++ headers
#include <sstream>   // msg
#include <iostream>  // cout
#include <stdexcept> // runtime erro
#include <stdio.h>   // fopen and fwrite

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp" 
#include "cr.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../globals.hpp"
#include "../coordinates/coordinates.hpp"
#include "integrators/cr_integrators.hpp"

// Default Diffusion Function
void DefaultDiff(MeshBlock *pmb, AthenaArray<Real> &u_cr, 
                 AthenaArray<Real> &prim, AthenaArray<Real> &bcc, 
		 Real dt) {
  CosmicRay *pcr=pmb->pcr;
  
  int il=pmb->is-1, iu=pmb->ie+1;
  int jl=pmb->js, ju=pmb->je;
  if (pmb->block_size.nx2 > 1) {jl -= 1; ju += 1;}
  int kl=pmb->ks, ku=pmb->ke;
  if (pmb->block_size.nx3 > 1) {kl -= 1; ku += 1;}

  // Initialize sigma_diff to maximum opacity everywhere
  for(int k=kl; k<=ku; ++k){
    for(int j=jl; j<=ju; ++j){
#pragma omp simd
      for(int i=il; i<=iu; ++i){
        pcr->sigma_diff(0,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;  
      }
    }
  }
  
  // Calculate B dot Grad Pc
  for(int k=kl; k<=ku; ++k){
    for(int j=jl; j<=ju; ++j){
      // Diffusion coefficient is calculated with respect to B direction
      // Use a simple estimate of Grad Pc

      // x component
      pmb->pcoord->CenterWidth1(k,j,il-1,iu+1,pcr->cwidth);
      for(int i=il; i<=iu; ++i){
        Real distance = 0.5*(pcr->cwidth(i-1) + pcr->cwidth(i+1))
                       + pcr->cwidth(i);
        Real dprdx=(pcr->prtensor_cr(PC11,k,j,i+1) * u_cr(CRE,k,j,i+1)
                     - pcr->prtensor_cr(PC11,k,j,i-1) * u_cr(CRE,k,j,i-1));
        dprdx /= distance;
        pcr->sigma_adv(0,k,j,i) = dprdx;
      }
      // y component
      pmb->pcoord->CenterWidth2(k,j-1,il,iu,pcr->cwidth1);       
      pmb->pcoord->CenterWidth2(k,j,il,iu,pcr->cwidth);
      pmb->pcoord->CenterWidth2(k,j+1,il,iu,pcr->cwidth2);

      for(int i=il; i<=iu; ++i){
        Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                       + pcr->cwidth(i);
        Real dprdy=(pcr->prtensor_cr(PC22,k,j+1,i) * u_cr(CRE,k,j+1,i)
                         - pcr->prtensor_cr(PC22,k,j-1,i) * u_cr(CRE,k,j-1,i));
        dprdy /= distance;
        pcr->sigma_adv(1,k,j,i) = dprdy;
      } 
      // z component
      pmb->pcoord->CenterWidth3(k-1,j,il,iu,pcr->cwidth1);       
      pmb->pcoord->CenterWidth3(k,j,il,iu,pcr->cwidth);
      pmb->pcoord->CenterWidth3(k+1,j,il,iu,pcr->cwidth2);

      for(int i=il; i<=iu; ++i){
        Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                        + pcr->cwidth(i);
        Real dprdz=(pcr->prtensor_cr(PC33,k+1,j,i) * u_cr(CRE,k+1,j,i)
                         - pcr->prtensor_cr(PC33,k-1,j,i) * u_cr(CRE,k-1,j,i));
        dprdz /= distance;
        pcr->sigma_adv(2,k,j,i) = dprdz;
      }       

      // Calculate the rotation matrix 
      // We need this to determine the direction of rotation velocity
      // The information stored in the array b_angle as follows:
      // b_angle[0]=sin_theta_b
      // b_angle[1]=cos_theta_b
      // b_angle[2]=sin_phi_b
      // b_angle[3]=cos_phi_b
      for(int i=il; i<=iu; ++i){
        Real bxby = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                         bcc(IB2,k,j,i)*bcc(IB2,k,j,i));
        Real btot = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                         bcc(IB2,k,j,i)*bcc(IB2,k,j,i) + 
                         bcc(IB3,k,j,i)*bcc(IB3,k,j,i));
            
        if(btot > TINY_NUMBER){
          pcr->b_angle(0,k,j,i) = bxby/btot;
          pcr->b_angle(1,k,j,i) = bcc(IB3,k,j,i)/btot;
        }else{
          pcr->b_angle(0,k,j,i) = 1.0;
          pcr->b_angle(1,k,j,i) = 0.0;
        }
        if(bxby > TINY_NUMBER){
          pcr->b_angle(2,k,j,i) = bcc(IB2,k,j,i)/bxby;
          pcr->b_angle(3,k,j,i) = bcc(IB1,k,j,i)/bxby;
        }else{
          pcr->b_angle(2,k,j,i) = 0.0;
          pcr->b_angle(3,k,j,i) = 1.0;            
        }

       	Real va = sqrt(btot*btot/prim(IDN,k,j,i));
        if(va < TINY_NUMBER){
          pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
        }else{
          Real b_grad_pc = bcc(IB1,k,j,i) * pcr->sigma_adv(0,k,j,i)
                         + bcc(IB2,k,j,i) * pcr->sigma_adv(1,k,j,i)
                         + bcc(IB3,k,j,i) * pcr->sigma_adv(2,k,j,i);
          pcr->sigma_adv(0,k,j,i) = fabs(b_grad_pc)/(btot * va * (1.0 + 
                               pcr->prtensor_cr(PC11,k,j,i)) * (1.0/pcr->vmax) * 
                               u_cr(CRE,k,j,i));
        }
         
	pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;

      }//end i        
    }// end j
  }// end k
} 

// Set the default CR Pressure Tensor to be isotropic
void DefaultCRTensor(MeshBlock *pmb, AthenaArray<Real> &prim) {
  CosmicRay *pcr=pmb->pcr;
  
  int nz1 = pmb->block_size.nx1 + 2*(NGHOST);
  int nz2 = pmb->block_size.nx2;
  if(nz2 > 1) nz2 += 2*(NGHOST);
  int nz3 = pmb->block_size.nx3;
  if(nz3 > 1) nz3 += 2*(NGHOST);

  for(int k=0; k<nz3; ++k){
    for(int j=0; j<nz2; ++j){
      for(int i=0; i<nz1; ++i){
        // Isotropic Pressure of 1/3
	pcr->prtensor_cr(PC11,k,j,i) = 1.0/3.0;
        pcr->prtensor_cr(PC22,k,j,i) = 1.0/3.0;
        pcr->prtensor_cr(PC33,k,j,i) = 1.0/3.0;
        pcr->prtensor_cr(PC12,k,j,i) = 0.0;
        pcr->prtensor_cr(PC13,k,j,i) = 0.0;
        pcr->prtensor_cr(PC23,k,j,i) = 0.0;
      }
    }
  }
}

// Constructor
CosmicRay::CosmicRay(MeshBlock *pmb, ParameterInput *pin) {
  // Set variables from input
  vmax = pin->GetOrAddReal("cr","vmax",1.0);
  vlim = pin->GetOrAddReal("cr","vlim",0.9);
  max_opacity = pin->GetOrAddReal("cr","max_opacity",1.e10);
  
  // Pointer to MeshBlock containing this fluid
  pmy_block = pmb;

  // Calculate number of zones in each dimension (Block size + Twice number of ghost zones)
  int n1z = pmy_block->block_size.nx1 + 2*(NGHOST);
  int n2z = (pmy_block->block_size.nx2 > 1) ? (pmy_block->block_size.nx2 + 2*(NGHOST)) : 1 ;
  int n3z = (pmy_block->block_size.nx3 > 1) ? (pmy_block->block_size.nx3 + 2*(NGHOST)) : 1 ;
  
  // Create arrays for cosmic ray energy density and flux
  u_cr.NewAthenaArray(NCR,n3z,n2z,n1z);
  u_cr1.NewAthenaArray(NCR,n3z,n2z,n1z);
  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time","integrator","vl2");
  if (integrator == "ssprk5_4")
    // future extension may add "int nregister" to Hydro class
    u_cr2.NewAthenaArray(NCR,n3z,n2z,n1z);

  sigma_diff.NewAthenaArray(3,n3z,n2z,n1z);
  sigma_adv.NewAthenaArray(3,n3z,n2z,n1z);

  v_adv.NewAthenaArray(3,n3z,n2z,n1z);
  v_diff.NewAthenaArray(3,n3z,n2z,n1z);

  prtensor_cr.NewAthenaArray(6,n3z,n2z,n1z);

  b_grad_pc.NewAthenaArray(n3z,n2z,n1z);
  b_angle.NewAthenaArray(4,n3z,n2z,n1z);
  
  // Allocate memory to store the transport flux
  flux[X1DIR].NewAthenaArray(NCR,n3z,n2z,n1z);
  if(n2z > 1) flux[X2DIR].NewAthenaArray(NCR,n3z,n2z,n1z);
  if(n3z > 1) flux[X3DIR].NewAthenaArray(NCR,n3z,n2z,n1z);

  cwidth.NewAthenaArray(n1z);
  cwidth1.NewAthenaArray(n1z);
  cwidth2.NewAthenaArray(n1z);
  
  // Set Default Functions for Diffusion and CRTensor
  UpdateDiff = DefaultDiff;
  UpdateCRTensor = DefaultCRTensor;

  // Create a Integrator object
  pcrintegrator = new CRIntegrator(this, pin);
}

// Destructor
CosmicRay::~CosmicRay() {
  u_cr.DeleteAthenaArray();
  u_cr1.DeleteAthenaArray();
  u_cr2.DeleteAthenaArray();

  sigma_diff.DeleteAthenaArray();
  sigma_adv.DeleteAthenaArray();
  prtensor_cr.DeleteAthenaArray();
  b_grad_pc.DeleteAthenaArray();
  b_angle.DeleteAthenaArray();

  v_adv.DeleteAthenaArray();
  v_diff.DeleteAthenaArray();

  cwidth.DeleteAthenaArray();
  cwidth1.DeleteAthenaArray();
  cwidth2.DeleteAthenaArray();
  
  flux[X1DIR].DeleteAthenaArray();
  if(pmy_block->block_size.nx2 > 1) flux[X2DIR].DeleteAthenaArray();
  if(pmy_block->block_size.nx3 > 1) flux[X3DIR].DeleteAthenaArray();
  
  delete pcrintegrator;
}

// Enroll the Diffusion Function
void CosmicRay::EnrollDiffFunction(CROpa_t MyDiffFunction) {
  UpdateDiff = MyDiffFunction;
}

// Enroll the CosmicRay Tensor Function
void CosmicRay::EnrollCRTensorFunction(CR_t MyTensorFunction) {
  UpdateCRTensor = MyTensorFunction;
}
