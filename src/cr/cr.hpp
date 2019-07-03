#ifndef CR_HPP
#define CR_HPP

#include <memory>
#include "../athena.hpp"
#include "../athena_arrays.hpp"

// Forward Declarations
class MeshBlock;
class ParameterInput;
class CRIntegrator;

// CosmicRay data structure and methods
class CosmicRay {
  public:
  // Constructor/Destructor
  CosmicRay(MeshBlock *pmb, ParameterInput *pin);
  ~CosmicRay();

  // CR Energy Density
  AthenaArray<Real> u_cr;

  // CR Flux Tensor (In units of (1/vmax))
  AthenaArray<Real> flux[3];

  // CR Pressure Tensor
  AthenaArray<Real> prtensor_cr;

  // Diffusion coefficients for normal diffusion and advection terms
  // In units of (1/vmax)
  AthenaArray<Real> sigma_diff, sigma_adv;

  // Streaming and diffusion velocities
  AthenaArray<Real> v_diff, v_adv;

  // Limits
  Real vmax; // Effective speed of light
  Real max_opacity;

  // Constant sigma_diffusion
  Real sigma_diffusion; // in units of (1/vmax)

  // Pointer to Mesh Block
  MeshBlock *pmy_block;

  // Pointer to CRIntegrator
  std::unique_ptr<CRIntegrator> pcrintegrator;

  // Function Pointers to user defined Diffusion and CR Tensor Functions
  CRDiff_t UpdateDiff;
  CRTensor_t UpdateCRTensor;

  // Enroll a user-defined Diffusion Function
  void EnrollDiffFunction(CRDiff_t MyDiffFunction);

  // Enroll a user-defined CR Tensor function
  void EnrollCRTensorFunction(CRTensor_t MyTensorFunction);

  // Extra CR energy density arrays are for use during time integration
  AthenaArray<Real> u_cr1, u_cr2;

  // Arrays to store intermediate results
  AthenaArray<Real> cwidth;
  AthenaArray<Real> cwidth1;
  AthenaArray<Real> cwidth2;
  AthenaArray<Real> b_grad_pc;
  AthenaArray<Real> b_angle; // sin(theta),cos(theta),sin(phi),cos(phi)
};

#endif // CR_HPP
