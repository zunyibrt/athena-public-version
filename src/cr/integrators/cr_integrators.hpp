#ifndef CRINTEGRATORS_HPP
#define CRINTEGRATORS_HPP

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../cr.hpp"

// Forward Declarations
class MeshBlock;
class ParameterInput;
class CosmicRay;

// Data Structure for CR Integrator
class CRIntegrator {
  public:
  CRIntegrator(CosmicRay *pcr, ParameterInput *pin);
  ~CRIntegrator();

  CosmicRay *pmy_cr;

  // Functions called by the time integrator in order
  // cr_transport.cpp
  void CalculateFluxes(MeshBlock *pmb, AthenaArray<Real> &w,
                       AthenaArray<Real> &bcc, AthenaArray<Real> &u_cr,
                       int reconstruct_order);

  void WeightedAveU(MeshBlock* pmb, AthenaArray<Real> &u_out,
         		        AthenaArray<Real> &u_in1, AthenaArray<Real> &u_in2,
         		        Real const wght[3]);

  void AddFluxDivergenceToAverage(MeshBlock *pmb, AthenaArray<Real> &u_cr,
                                  Real const wght);

  // cr_source.cpp
  void AddSourceTerms(MeshBlock *pmb, Real const dt, AthenaArray<Real> &u,
                      AthenaArray<Real> &w, AthenaArray<Real> &u_cr);

  private:
  // Helper functions
  // Not associated with main functions (cr.cpp)
  void RotateVec(Real const sint, Real const ost,
                 Real const sinp, Real const cosp,
                 Real &v1, Real &v2, Real &v3);

  void InvRotateVec(Real const sint, Real const cost,
                    Real const sinp, Real const cosp,
                    Real &v1, Real &v2, Real &v3);

  // cr_flux.cpp
  void CRFlux(int dir, int k, int j, int il, int iu, AthenaArray<Real> &w_l,
              AthenaArray<Real> &w_r, AthenaArray<Real> &vel_l,
	            AthenaArray<Real> &vel_r, AthenaArray<Real> &eddl,
	            AthenaArray<Real> &eddr, AthenaArray<Real> &vdiff_l,
	            AthenaArray<Real> &vdiff_r, AthenaArray<Real> &flx);

  // cr_reconstruction.cpp
  void DonorCellX1(int const k, int const j,
                   int const il, int const iu, AthenaArray<Real> const &u_cr,
                   AthenaArray<Real> const &prim, AthenaArray<Real> const &edd,
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
                   AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                   AthenaArray<Real> &eddl, AthenaArray<Real> &eddr);

  void DonorCellX2(int const k, int const j,
                   int const il, int const iu, AthenaArray<Real> const &u_cr,
                   AthenaArray<Real> const &prim, AthenaArray<Real> const &edd,
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
                   AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                   AthenaArray<Real> &eddl, AthenaArray<Real> &eddr);

  void DonorCellX3(int const k, int const j,
                   int const il, int const iu, AthenaArray<Real> const &u_cr,
                   AthenaArray<Real> const &prim, AthenaArray<Real> const &edd,
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
                   AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                   AthenaArray<Real> &eddl, AthenaArray<Real> &eddr);

  void PieceWiseLinear(int const k, int const j,
                       int const il, int const iu, AthenaArray<Real> &u_cr,
                       AthenaArray<Real> &prim, AthenaArray<Real> &edd,
                       AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
                       AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                       AthenaArray<Real> &eddl, AthenaArray<Real> &eddr,
		                   int dir);

  void GetOneVariableX1(int const k, int const j, int const il, int const iu,
		                    AthenaArray<Real> const &q, AthenaArray<Real> &ql,
			                  AthenaArray<Real> &qr);

  void GetOneVariableX2(int const k, int const j,
                        int const il, int const iu, AthenaArray<Real> const &q,
                        AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void GetOneVariableX3(int const k, int const j,
                        int const il, int const iu, AthenaArray<Real> const &q,
                        AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  // Internal Arrays
  Real taufact_;
  int vel_flx_flag_;
  AthenaArray<Real> flx_;
  AthenaArray<Real> vel_l_,vel_r_,wl_,wr_,vdiff_l_,vdiff_r_;
  AthenaArray<Real> eddl_,eddr_;
  AthenaArray<Real> grad_pc_;
  AthenaArray<Real> ec_source_;
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_, cwidth_;

};

#endif // CRINTEGRATORS_HPP
