#ifndef TCINTEGRATORS_HPP
#define TCINTEGRATORS_HPP

// Athena++ classes headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../task_list/task_list.hpp"

// Forward Declarations
class Hydro;
class ParameterInput;
class ThermalConduction;

// Data Structure for TC Integrator
class TCIntegrator {
public:
  TCIntegrator(ThermalConduction *ptc, ParameterInput *pin);
  ~TCIntegrator();

  ThermalConduction *pmy_tc;

  // Functions called by the time integrator in order
  // tc_transport.cpp
  void CalculateFluxes(MeshBlock *pmb,
                       AthenaArray<Real> &w, AthenaArray<Real> &bcc,
                       AthenaArray<Real> &u_tc, int reconstruct_order);

  void WeightedAveU(MeshBlock* pmb, AthenaArray<Real> &u_out,
         		        AthenaArray<Real> &u_in1, AthenaArray<Real> &u_in2,
         		        Real const weights[3]);

  void AddFluxDivergenceToAverage(MeshBlock *pmb, AthenaArray<Real> &u_tc,
                                  Real const weight);

  void AddSourceTerms(MeshBlock *pmb, Real const dt,
                      AthenaArray<Real> &u, AthenaArray<Real> &u_tc);

private:
  // Helper functions
  // Not associated with main functions (cr.cpp)
  void RotateVec(Real const sint, Real const ost,
                 Real const sinp, Real const cosp,
                 Real &v1, Real &v2, Real &v3);

  void InvRotateVec(Real const sint, Real const cost,
                    Real const sinp, Real const cosp,
                    Real &v1, Real &v2, Real &v3);

  // tc_flux.cpp
  void TCFlux(int fdir, int il, int iu,
              AthenaArray<Real> &t_l,     AthenaArray<Real> &t_r,
              AthenaArray<Real> &rho_l,   AthenaArray<Real> &rho_r,
              AthenaArray<Real> &w_l,     AthenaArray<Real> &w_r,
              AthenaArray<Real> &vdiff_l, AthenaArray<Real> &vdiff_r,
              AthenaArray<Real> &flx);

  // tc_reconstruction.cpp
  void DonorCellX1(int const k, int const j, int const il, int const iu,
                   AthenaArray<Real> const &u_tc,
                   AthenaArray<Real> const &rho,
                   AthenaArray<Real> const &tgas,
                   AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
                   AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r);

  void DonorCellX2(int const k, int const j, int const il, int const iu,
                   AthenaArray<Real> const &u_tc,
                   AthenaArray<Real> const &rho,
                   AthenaArray<Real> const &tgas,
                   AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
                   AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r);

  void DonorCellX3(int const k, int const j, int const il, int const iu,
                   AthenaArray<Real> const &u_tc,
                   AthenaArray<Real> const &rho,
                   AthenaArray<Real> const &tgas,
                   AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
                   AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r);

  void PieceWiseLinear(int const k, int const j, int const il, int const iu,
                       AthenaArray<Real> &u_tc,
                       AthenaArray<Real> &rho,
                       AthenaArray<Real> &tgas,
                       AthenaArray<Real> &rho_l, AthenaArray<Real> &rho_r,
                       AthenaArray<Real> &t_l, AthenaArray<Real> &t_r,
                       AthenaArray<Real> &w_l, AthenaArray<Real> &w_r,
		                   int dir);

  void GetOneVariableX1(int const k, int const j, int const il, int const iu,
		                    AthenaArray<Real> const &q,
                        AthenaArray<Real> &ql,
			                  AthenaArray<Real> &qr);

  void GetOneVariableX2(int const k, int const j, int const il, int const iu,
                        AthenaArray<Real> const &q,
                        AthenaArray<Real> &ql,
                        AthenaArray<Real> &qr);

  void GetOneVariableX3(int const k, int const j, int const il, int const iu,
                        AthenaArray<Real> const &q,
                        AthenaArray<Real> &ql,
                        AthenaArray<Real> &qr);

  // Internal Arrays
  AthenaArray<Real> flx_;
  AthenaArray<Real> wl_,wr_,vdiff_,vdiff_l_,vdiff_r_;
  AthenaArray<Real> rho_l_, rho_r_, tgas_l_, tgas_r_;
  AthenaArray<Real> tc_esource_;
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_, cwidth_;

};

#endif // TCINTEGRATORS_HPP
