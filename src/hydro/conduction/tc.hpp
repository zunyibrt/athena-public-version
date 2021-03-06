#ifndef TC_HPP
#define TC_HPP

// Athena++ classes headers
#include <memory>
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"

// Forward Declarations
class MeshBlock;
class ParameterInput;
class TCIntegrator;

// Prototype for user-defined conduction coefficient
typedef void (*TCkappa_t)(MeshBlock *pmb,
                          AthenaArray<Real> &prim,
                          AthenaArray<Real> &bcc);

// ThermalConduction data structure and methods
class ThermalConduction {
  friend class TCIntegrator;

public:
  // Constructor/Destructor
  ThermalConduction(Hydro *phydro, ParameterInput *pin);
  ~ThermalConduction();

  // Arrays storing thermal conduction variables
  // Extra arrays are for use during time integration
  AthenaArray<Real> u_tc, u_tc1, u_tc2;

  // Store the flux during TC Transport, also needed for refinement
  AthenaArray<Real> flux[3];

  // Three components of conduction coefficients
  AthenaArray<Real> tc_kappa;
  AthenaArray<Real> b_angle;
  AthenaArray<Real> rho, tgas;

  // Limits
  Real vmax; // the maximum velocity (effective speed of light)
  Real kappa_iso;
  Real kappa_aniso;

  // Pointer to parent hydro
  Hydro *pmy_hydro;

  // Pointer to Integrator
  std::unique_ptr<TCIntegrator> ptcintegrator;

  // The function pointer for the diffusion coefficient
  TCkappa_t UpdateKappa;

  //  Enroll a user-defined opacity function in problem generators
  void EnrollKappaFunction(TCkappa_t MyConductionFunction);
  void Initialize(MeshBlock *pmb, AthenaArray<Real> &prim,
                                  AthenaArray<Real> &tc_u);

};

#endif // TC_HPP
