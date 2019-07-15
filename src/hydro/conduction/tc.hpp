#ifndef NTC_HPP
#define NTC_HPP
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class
//======================================================================================

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include <string>

class MeshBlock;
class ParameterInput;
class TCIntegrator;

//! \class ThermalConduction
//  \brief Thermal Conduction data and functions


// prototype for user-defined conduction coefficient
typedef void (*TCkappa_t)(MeshBlock *pmb, 
                          AthenaArray<Real> &prim, 
                          AthenaArray<Real> &bcc);

// Array indices for  moments

class ThermalConduction {
  friend class TCIntegrator;
public:
  ThermalConduction(Hydro *phydro, ParameterInput *pin);
  ~ThermalConduction();
    
  AthenaArray<Real> u_tc, u_tc1; //thermal conduction flux

  //  three components of conduction coefficients
  AthenaArray<Real> tc_kappa; 
  AthenaArray<Real> b_angle;
  AthenaArray<Real> rho, tgas;

  AthenaArray<Real> flux[3]; // store transport flux, also need for refinement
 

  Real vmax; // the maximum velocity (effective speed of light)
  Real min_kappa;


  Hydro* pmy_hydro;    // ptr to the parent hydro

  NTCIntegrator *ptcintegrator;
  
  //Function in problem generators to update opacity
  void EnrollKappaFunction(TCkappa_t MyConductionFunction);
  void Initialize(MeshBlock *pmb, 
                  AthenaArray<Real> &prim, 
                  AthenaArray<Real> &tc_u);

  // The function pointer for the diffusion coefficient
  TCkappa_t UpdateKappa;

private:

};

#endif // NTC_HPP
