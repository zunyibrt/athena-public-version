// C++ headers
#include <sstream>  // msg
#include <iostream>  // cout
#include <stdexcept> // runtime erro
#include <stdio.h>  // fopen and fwrite
#include <memory>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../hydro.hpp"
#include "../../eos/eos.hpp"
#include "tc.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../../globals.hpp"
#include "../../coordinates/coordinates.hpp"
#include "integrators/tc_integrators.hpp"

// Default Conduction Coefficient Function
// Sets the default conductivity to be a small value in the default hydro case
// Also initializes rho and T from prim variables
inline void DefaultKappa(MeshBlock *pmb,
                         AthenaArray<Real> &prim,
                         AthenaArray<Real> &bcc)
{
  auto ptc = pmb->phydro->ptc;

  int kl = pmb->ks,   ku = pmb->ke;
  int jl = pmb->js,   ju = pmb->je;
  int il = pmb->is-1, iu = pmb->ie+1;
  if (pmb->block_size.nx2 > 1) {jl -= 1; ju += 1;}
  if (pmb->block_size.nx3 > 1) {kl -= 1; ku += 1;}

  // Initialize tc_kappa with user defined conductivities
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        ptc->tc_kappa(0,k,j,i) = ptc->kappa_iso + ptc->kappa_aniso;
        ptc->tc_kappa(1,k,j,i) = ptc->kappa_iso;
        ptc->tc_kappa(2,k,j,i) = ptc->kappa_iso;
      }
    }
  }

  // Diffusion coefficient is calculated with respect to B direction
  // Here we calculate the angles of B
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real bxby = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i));
          Real btot = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i) +
                           bcc(IB3,k,j,i)*bcc(IB3,k,j,i));

          if (btot > TINY_NUMBER) {
            ptc->b_angle(0,k,j,i) = bxby/btot;
            ptc->b_angle(1,k,j,i) = bcc(IB3,k,j,i)/btot;
          } else {
            ptc->b_angle(0,k,j,i) = 1.0;
            ptc->b_angle(1,k,j,i) = 0.0;
          }
          if (bxby > TINY_NUMBER) {
            ptc->b_angle(2,k,j,i) = bcc(IB2,k,j,i)/bxby;
            ptc->b_angle(3,k,j,i) = bcc(IB1,k,j,i)/bxby;
          } else {
            ptc->b_angle(2,k,j,i) = 0.0;
            ptc->b_angle(3,k,j,i) = 1.0;
          }

        }//end i
      }// end j
    }// end k
  }// end MHD

}

// Constructor
ThermalConduction::ThermalConduction(Hydro *phydro, ParameterInput *pin)
{
  vmax = pin->GetOrAddReal("tc","vmax",1.0);
  std::string integrator = pin->GetOrAddString("time","integrator","vl2");
  kappa_aniso = pin->GetOrAddReal("tc","kappa_aniso",1.e-10);
  kappa_iso = pin->GetOrAddReal("tc","kappa_iso",1.e-10);

  // Pointer to parent hydro
  pmy_hydro = phydro;

  // Calculate number of zones in each dimension
  // (Block size + Twice number of ghost zones)
  int n1z = pmy_hydro->pmy_block->block_size.nx1 + 2*NGHOST;
  int n2z = pmy_hydro->pmy_block->block_size.nx2;
  int n3z = pmy_hydro->pmy_block->block_size.nx3;
  if (n2z > 1) {n2z += 2*NGHOST;}
  if (n3z > 1) {n2z += 2*NGHOST;}

  // The four  stored conserved variables are
  // e, (T/e)F_c1, (T/e)F_c2, (T/e)F_c3
  u_tc.NewAthenaArray(4,n3z,n2z,n1z);
  u_tc1.NewAthenaArray(4,n3z,n2z,n1z);
  u_tc2.NewAthenaArray(4,n3z,n2z,n1z);

  tc_kappa.NewAthenaArray(3,n3z,n2z,n1z);
  b_angle.NewAthenaArray(4,n3z,n2z,n1z);
  rho.NewAthenaArray(n3z,n2z,n1z);
  tgas.NewAthenaArray(n3z,n2z,n1z);

  // Allocate memory to store the transport flux
  flux[X1DIR].NewAthenaArray(4,n3z,n2z,n1z);
  if (n2z > 1) {flux[X2DIR].NewAthenaArray(4,n3z,n2z,n1z);}
  if (n3z > 1) {flux[X3DIR].NewAthenaArray(4,n3z,n2z,n1z);}

  // set a default opacity function
  UpdateKappa = DefaultKappa;

  // Create a Integrator object
  ptcintegrator = std::make_unique<TCIntegrator>(this, pin);
}

// Destructor
ThermalConduction::~ThermalConduction()
{
  u_tc.DeleteAthenaArray();
  u_tc1.DeleteAthenaArray();
  u_tc2.DeleteAthenaArray();

  tc_kappa.DeleteAthenaArray();
  b_angle.DeleteAthenaArray();
  rho.DeleteAthenaArray();
  tgas.DeleteAthenaArray();

  flux[X1DIR].DeleteAthenaArray();
  if(pmy_hydro->pmy_block->block_size.nx2 > 1) flux[X2DIR].DeleteAthenaArray();
  if(pmy_hydro->pmy_block->block_size.nx3 > 1) flux[X3DIR].DeleteAthenaArray();
}

// Enroll the function to update opacity
void ThermalConduction::EnrollKappaFunction(TCkappa_t MyKappaFunction)
{
  UpdateKappa = MyKappaFunction;
}

void ThermalConduction::Initialize(MeshBlock *pmb, AthenaArray<Real> &prim,
	                                                 AthenaArray<Real> &tc_u)
{
  Real gamma_1=pmb->peos->GetGamma()-1.0;

  int n1 = pmb->block_size.nx1 + 2*NGHOST;
  int n2 = pmb->block_size.nx2;
  int n3 = pmb->block_size.nx3;
  if (n2 > 1) {n2 += 2*NGHOST;}
  if (n3 > 1) {n3 += 2*NGHOST;}

  for (int k=0; k<n3; ++k){
    for(int j=0; j<n2; ++j){
      for(int i=0; i<n1; ++i){
        tgas(k,j,i) = prim(IPR,k,j,i)/prim(IDN,k,j,i);
        // rho here actually should be e/T=pgas/(T(gamma-1))
        rho(k,j,i) = prim(IDN,k,j,i)/gamma_1;
        // Set the first conserved variable to be the gas internal energy
        tc_u(0,k,j,i) = prim(IPR,k,j,i)/gamma_1;
      }
    }
  }


}
