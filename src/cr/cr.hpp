#ifndef CR_HPP
#define CR_HPP

// C++ Header
#include <string>

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

// Forward Declarations
class MeshBlock;
class ParameterInput;
class CRIntegrator;

// Prototype for user-defined diffusion coefficient
typedef void (*CR_t)(MeshBlock *pmb, AthenaArray<Real> &prim);
typedef void (*CROpa_t)(MeshBlock *pmb, AthenaArray<Real> &u_cr,
                        AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
			                  Real dt);

// Array indices for moments
enum {CRE=0, CRF1=1, CRF2=2, CRF3=3};
// Array indices for Pressure Tensor
enum {PC11=0, PC22=1, PC33=2, PC12=3, PC13=4, PC23=5};

// CosmicRay data structure and methods
class CosmicRay {
  friend class CRIntegrator;

  public:
  // Constructor/Destructor
  CosmicRay(MeshBlock *pmb, ParameterInput *pin);
  ~CosmicRay();

  // Arrays storing CR energy densities
  // Extra arrays are for use during time integration
  AthenaArray<Real> u_cr, u_cr1, u_cr2;

  // Store the flux during CR Transport, also needed for refinement
  AthenaArray<Real> flux[3];

  // Cosmic Ray Pressure Tensor
  AthenaArray<Real> prtensor_cr;

  // Diffusion coefficients for normal diffusion and advection terms
  // In units of (1/vmax)
  AthenaArray<Real> sigma_diff, sigma_adv;

  // Streaming and diffusion velocities
  AthenaArray<Real> v_diff, v_adv;

  // Velocity limits
  Real vmax; // Effective speed of light
  Real max_opacity;
  Real sigma_diffusion; // in units of (1/vmax)

  // Pointer to Mesh Block
  MeshBlock* pmy_block;

  // Pointer to Integrator
  CRIntegrator *pcrintegrator;

  // Enroll a user-defined diffusion function in problem generators
  void EnrollDiffFunction(CROpa_t MyDiffFunction);

  // Enroll a user-defined CR tensor function
  void EnrollCRTensorFunction(CR_t MyTensorFunction);

  // Functions for updating the Diffusion and CR Tensor
  CROpa_t UpdateDiff;
  CR_t UpdateCRTensor;

  // Arrays to store intermediate results
  AthenaArray<Real> cwidth;
  AthenaArray<Real> cwidth1;
  AthenaArray<Real> cwidth2;
  AthenaArray<Real> b_grad_pc; // array to store B dot Grad Pc
  AthenaArray<Real> b_angle;   // sin(theta),cos(theta),sin(phi),cos(phi)
                               // of B direction
};

#endif
