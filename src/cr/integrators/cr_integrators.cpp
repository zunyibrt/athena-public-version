#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../cr.hpp"
#include "cr_integrators.hpp"

// Constructor for CRIntegrator
CRIntegrator::CRIntegrator(CosmicRay *pcr, ParameterInput *pin)
{
  pmy_cr = pcr;

  int nthreads = pcr->pmy_block->pmy_mesh->GetNumMeshThreads();
  int ncells1 = pcr->pmy_block->block_size.nx1 + 2*(NGHOST);
  int ncells2 = pcr->pmy_block->block_size.nx2;
  if (ncells2 > 1) ncells2 += 2*(NGHOST);
  int ncells3 = pcr->pmy_block->block_size.nx3;
  if (ncells3 > 1) ncells3 += 2*(NGHOST);

  flx_.NewAthenaArray(nthreads,NCR,ncells1);
  vel_l_.NewAthenaArray(nthreads,ncells1);
  vel_r_.NewAthenaArray(nthreads,ncells1);
  vdiff_l_.NewAthenaArray(nthreads,ncells1);
  vdiff_r_.NewAthenaArray(nthreads,ncells1);
  crptl_.NewAthenaArray(nthreads,6,ncells1);
  crptr_.NewAthenaArray(nthreads,6,ncells1);
  wl_.NewAthenaArray(nthreads,(NCR),ncells1);
  wr_.NewAthenaArray(nthreads,(NCR),ncells1);

  cwidth_.NewAthenaArray(nthreads,ncells1);

  x1face_area_.NewAthenaArray(nthreads,ncells1+1);
  if (ncells2 > 1) {
    x2face_area_.NewAthenaArray(nthreads,ncells1);
    x2face_area_p1_.NewAthenaArray(nthreads,ncells1);
  }
  if (ncells3 > 1) {
    x3face_area_.NewAthenaArray(nthreads,ncells1);
    x3face_area_p1_.NewAthenaArray(nthreads,ncells1);
  }
  cell_volume_.NewAthenaArray(nthreads,ncells1);

  grad_pc_.NewAthenaArray(3,ncells3,ncells2,ncells1);
  ec_source_.NewAthenaArray(ncells3,ncells2,ncells1);
}

// Destructor
CRIntegrator::~CRIntegrator()
{
  flx_.DeleteAthenaArray();
  vel_l_.DeleteAthenaArray();
  vel_r_.DeleteAthenaArray();
  vdiff_l_.DeleteAthenaArray();
  vdiff_r_.DeleteAthenaArray();
  crptl_.DeleteAthenaArray();
  crptr_.DeleteAthenaArray();
  wl_.DeleteAthenaArray();
  wr_.DeleteAthenaArray();

  cwidth_.DeleteAthenaArray();

  x1face_area_.DeleteAthenaArray();
  if (pmy_cr->pmy_block->block_size.nx2 > 1) {
    x2face_area_.DeleteAthenaArray();
    x2face_area_p1_.DeleteAthenaArray();
  }
  if (pmy_cr->pmy_block->block_size.nx3 > 1) {
    x3face_area_.DeleteAthenaArray();
    x3face_area_p1_.DeleteAthenaArray();
  }
  cell_volume_.DeleteAthenaArray();

  grad_pc_.DeleteAthenaArray();
  ec_source_.DeleteAthenaArray();
}

void CRIntegrator::RotateVec(Real const sint, Real const cost,
                             Real const sinp, Real const cosp,
                             Real &v1, Real &v2, Real &v3)
{
  // Applies the transformation R2*R1*v in place
  // The two rotation matrices are:
  // R_1 =                  R_2 =
  // [cos_p  sin_p 0]       [sin_t  0 cos_t]
  // [-sin_p cos_p 0]       [0      1    0]
  // [0       0    1]       [-cos_t 0 sin_t]

  // Apply R1
  Real newv1 =  cosp * v1 + sinp * v2;
  v2 = -sinp * v1 + cosp * v2;

  // Apply R2
  v1 =  sint * newv1 + cost * v3;
  Real newv3 = -cost * newv1 + sint * v3;
  v3 = newv3;
}

void CRIntegrator::InvRotateVec(Real const sint, Real const cost,
                                Real const sinp, Real const cosp,
                                Real &v1, Real &v2, Real &v3)
{
  // Applies the inverse transformation R1_inv*R2_inv*v in place
  // The two rotation matrices are:
  // R1_inv=                R_2_inv=
  // [cos_p  -sin_p 0]      [sin_t  0 -cos_t]
  // [sin_p cos_p 0]        [0      1    0]
  // [0       0    1]       [cos_t 0 sin_t]

  // Apply R2_inv
  Real newv1 = sint * v1 - cost * v3;
  v3 = cost * v1 + sint * v3;

  // Apply R1_inv
  v1 = cosp * newv1 - sinp * v2;
  Real newv2 = sinp * newv1 + cosp * v2;
  v2 = newv2;
}
