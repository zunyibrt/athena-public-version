#ifndef CR_HPP
#define CR_HPP
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class
//======================================================================================

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include <string>

// Forward Declarations
class MeshBlock;
class ParameterInput;
class CRIntegrator;

// prototype for user-defined diffusion coefficient
typedef void (*CR_t)(MeshBlock *pmb, AthenaArray<Real> &prim);
typedef void (*CROpa_t)(MeshBlock *pmb, AthenaArray<Real> &u_cr, 
                      AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Real dt);

// Array indices for  moments
enum {CRE=0, CRF1=1, CRF2=2, CRF3=3};
// Indices for Pressure Tensor
enum {PC11=0, PC22=1, PC33=2, PC12=3, PC13=4, PC23=5};

//! \class CosmicRay
//  \brief CosmicRay data and functions
class CosmicRay {
  friend class CRIntegrator;

  public:
  CosmicRay(MeshBlock *pmb, ParameterInput *pin);
  ~CosmicRay();
    
  AthenaArray<Real> u_cr, u_cr1; //cosmic ray energy density and flux

  //   diffusion coefficients for both normal diffusion term, and advection term
  AthenaArray<Real> sigma_diff, sigma_adv; 

  AthenaArray<Real> prtensor_cr; //   The cosmic ray pressure tensor

  AthenaArray<Real> v_adv; // streaming velocity
  AthenaArray<Real> v_diff; // the diffuion velocity, need to calculate the flux
    


  AthenaArray<Real> flux[3]; // store transport flux, also need for refinement
  
  Real vmax; // the maximum velocity (effective speed of light)
  Real vlim;
  Real max_opacity;


  MeshBlock* pmy_block;    // ptr to MeshBlock containing this Fluid
  
  CRIntegrator *pcrintegrator;
  
  
  //Function in problem generators to update opacity
  void EnrollDiffFunction(CROpa_t MyDiffFunction);


  //Function in problem generators to update opacity
  void EnrollCRTensorFunction(CR_t MyTensorFunction);

  // The function pointer for the diffusion coefficient
  CROpa_t UpdateDiff;
  CR_t UpdateCRTensor; 

  AthenaArray<Real> cwidth; 
  AthenaArray<Real> cwidth1;
  AthenaArray<Real> cwidth2;
  AthenaArray<Real> b_grad_pc; // array to store B\dot Grad Pc
  AthenaArray<Real> b_angle; //sin\theta,cos\theta,sin\phi,cos\phi of B direction
};
#endif 
