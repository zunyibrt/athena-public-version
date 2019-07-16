// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../parameter_input.hpp"
#include "../../../mesh/mesh.hpp"
#include "../tc.hpp"
#include "tc_integrators.hpp"
#include "../../hydro.hpp"

// Constructor for TCIntegrator
TCIntegrator::TCIntegrator(ThermalConduction *ptc, ParameterInput *pin)
{
  pmy_tc = ptc;

  taufact_ = pin->GetOrAddReal("tc","taucell",5.0);
  int nthreads = ptc->pmy_hydro->pmy_block->pmy_mesh->GetNumMeshThreads();
  int ncells1 = ptc->pmy_hydro->pmy_block->block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1;
  int ncells3 = 1;

  flx_.NewAthenaArray(nthreads,4,ncells1);
  vdiff_l_.NewAthenaArray(nthreads,ncells1);
  vdiff_r_.NewAthenaArray(nthreads,ncells1);
  rho_l_.NewAthenaArray(nthreads,ncells1);
  rho_r_.NewAthenaArray(nthreads,ncells1);
  tgas_l_.NewAthenaArray(nthreads,ncells1);
  tgas_r_.NewAthenaArray(nthreads,ncells1);
  wl_.NewAthenaArray(nthreads,4,ncells1);
  wr_.NewAthenaArray(nthreads,4,ncells1);

  cwidth_.NewAthenaArray(nthreads,ncells1);

  x1face_area_.NewAthenaArray(nthreads,ncells1+1);
  if(ptc->pmy_hydro->pmy_block->block_size.nx2 > 1) {
    x2face_area_.NewAthenaArray(nthreads,ncells1);
    x2face_area_p1_.NewAthenaArray(nthreads,ncells1);
    ncells2 = ptc->pmy_hydro->pmy_block->block_size.nx2 + 2*(NGHOST);

  }
  if(ptc->pmy_hydro->pmy_block->block_size.nx3 > 1) {
    x3face_area_.NewAthenaArray(nthreads,ncells1);
    x3face_area_p1_.NewAthenaArray(nthreads,ncells1);
    ncells3 = ptc->pmy_hydro->pmy_block->block_size.nx3 + 2*(NGHOST);

  }
  cell_volume_.NewAthenaArray(nthreads,ncells1);

  tc_esource_.NewAthenaArray(ncells3,ncells2,ncells1);
  vdiff_.NewAthenaArray(3,ncells3,ncells2,ncells1);
}

// destructor

TCIntegrator::~TCIntegrator()
{
  flx_.DeleteAthenaArray();
  vdiff_l_.DeleteAthenaArray();
  vdiff_r_.DeleteAthenaArray();

  wl_.DeleteAthenaArray();
  wr_.DeleteAthenaArray();
  rho_l_.DeleteAthenaArray();
  rho_r_.DeleteAthenaArray();
  tgas_l_.DeleteAthenaArray();
  tgas_r_.DeleteAthenaArray();
  tc_esource_.DeleteAthenaArray();
  vdiff_.DeleteAthenaArray();

  cwidth_.DeleteAthenaArray();

  x1face_area_.DeleteAthenaArray();
  if(pmy_tc->pmy_hydro->pmy_block->block_size.nx2 > 1) {
    x2face_area_.DeleteAthenaArray();
    x2face_area_p1_.DeleteAthenaArray();
  }
  if(pmy_tc->pmy_hydro->pmy_block->block_size.nx3 > 1) {
    x3face_area_.DeleteAthenaArray();
    x3face_area_p1_.DeleteAthenaArray();
  }
  cell_volume_.DeleteAthenaArray();
}

void TCIntegrator::RotateVec(const Real sint, const Real cost,
                             const Real sinp, const Real cosp,
                             Real &v1, Real &v2, Real &v3) {
  // vel1, vel2, vel3 are input
  // v1, v2, v3 are output
  // The two rotation matrices are:
  // R_1 =
  // [cos_p  sin_p 0]
  // [-sin_p cos_p 0]
  // [0       0    1]
  //
  // R_2 =
  // [sin_t  0 cos_t]
  // [0      1    0]
  // [-cos_t 0 sin_t]


  // First apply R1
  Real newv1 =  cosp * v1 + sinp * v2;
  v2 = -sinp * v1 + cosp * v2;

  // Then apply R2
  v1 =  sint * newv1 + cost * v3;
  Real newv3 = -cost * newv1 + sint * v3;
  v3 = newv3;
}

void TCIntegrator::InvRotateVec(const Real sint, const Real cost,
                                const Real sinp, const Real cosp,
                                Real &v1, Real &v2, Real &v3) {
  // vel1, vel2, vel3 are input
  // v1, v2, v3 are output
  // The two rotation matrices are:
  // R_1^-1=
  // [cos_p  -sin_p 0]
  // [sin_p cos_p 0]
  // [0       0    1]
  //
  // R_2^-1=
  // [sin_t  0 -cos_t]
  // [0      1    0]
  // [cos_t 0 sin_t]


  // First apply R2^-1
  Real newv1 = sint * v1 - cost * v3;
  v3 = cost * v1 + sint * v3;

  // Then apply R1^-1
  v1 = cosp * newv1 - sinp * v2;
  Real newv2 = sinp * newv1 + cosp * v2;
  v2 = newv2;
}
