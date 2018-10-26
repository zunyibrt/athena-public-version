#ifndef CRINTEGRATORS_HPP
#define CRINTEGRATORS_HPP

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../cr.hpp" 
#include "../../task_list/task_list.hpp"

// Forward Declarations
class MeshBlock;
class ParameterInput;
class CosmicRay;

// Data Structure for CR Integrator
class CRIntegrator {
  friend class CosmicRay;

  public:
  CRIntegrator(CosmicRay *pcr, ParameterInput *pin);
  ~CRIntegrator();
  
  CosmicRay *pmy_cr;

  void AddSourceTerms(MeshBlock *pmb, const Real dt, AthenaArray<Real> &u,
                      AthenaArray<Real> &w, AthenaArray<Real> &u_cr);

  void CalculateFluxes(MeshBlock *pmb, AthenaArray<Real> &w, 
		       AthenaArray<Real> &u_cr, int reconstruct_order);

  void AddFluxDivergenceToAverage(MeshBlock *pmb, AthenaArray<Real> &u_cr,
                                  AthenaArray<Real> &u, const Real wght,
                                  AthenaArray<Real> &w, AthenaArray<Real> &bcc);

  void WeightedAveU(MeshBlock* pmb, AthenaArray<Real> &u_out, 
		    AthenaArray<Real> &u_in1, AthenaArray<Real> &u_in2, 
		    const Real wght[3]);
  
  void CRFlux(int dir, int k, int j, int il, int iu, AthenaArray<Real> &w_l, 
              AthenaArray<Real> &w_r, AthenaArray<Real> &vel_l, 
	      AthenaArray<Real> &vel_r, AthenaArray<Real> &eddl, 
	      AthenaArray<Real> &eddr, AthenaArray<Real> &vdiff_l, 
	      AthenaArray<Real> &vdiff_r, AthenaArray<Real> &flx);

  void DonorCellX1(const int k, const int j,
                   const int il, const int iu, const AthenaArray<Real> &u_cr,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &edd, 
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r, 
                   AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                   AthenaArray<Real> &eddl, AthenaArray<Real> &eddr);   
                                  
  void DonorCellX2(const int k, const int j,
                   const int il, const int iu, const AthenaArray<Real> &u_cr,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &edd, 
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r, 
                   AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                   AthenaArray<Real> &eddl, AthenaArray<Real> &eddr);   

  void DonorCellX3(const int k, const int j,
                   const int il, const int iu, const AthenaArray<Real> &u_cr,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &edd,
                   AthenaArray<Real> &w_l, AthenaArray<Real> &w_r, 
                   AthenaArray<Real> &v_l, AthenaArray<Real> &v_r,
                   AthenaArray<Real> &eddl, AthenaArray<Real> &eddr);  

  void PieceWiseLinear(const int k, const int j,
                       const int il, const int iu, AthenaArray<Real> &u_cr,
                       AthenaArray<Real> &prim, AthenaArray<Real> &edd,
                       AthenaArray<Real> &w_l, AthenaArray<Real> &w_r, 
                       AthenaArray<Real> &v_l, AthenaArray<Real> &v_r, 
                       AthenaArray<Real> &eddl, AthenaArray<Real> &eddr, 
		       int dir);

  void GetOneVariableX1(const int k, const int j, const int il, const int iu, 
		        const AthenaArray<Real> &q, AthenaArray<Real> &ql, 
			AthenaArray<Real> &qr);

  void GetOneVariableX2(const int k, const int j, 
                        const int il, const int iu, const AthenaArray<Real> &q, 
                        AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void GetOneVariableX3(const int k, const int j, 
                        const int il, const int iu, const AthenaArray<Real> &q, 
                        AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void RotateVec(const Real sint, const Real cost, 
                 const Real sinp, const Real cosp, 
                 Real &v1, Real &v2, Real &v3);

  void InvRotateVec(const Real sint, const Real cost, 
                    const Real sinp, const Real cosp, 
                    Real &v1, Real &v2, Real &v3);

  private:
  AthenaArray<Real> flx_;
  AthenaArray<Real> vel_l_,vel_r_,wl_,wr_,vdiff_l_,vdiff_r_;
  AthenaArray<Real> eddl_,eddr_;
  AthenaArray<Real> grad_pc_;

  // Temporary array to store the flux
  Real taufact_;
  int vel_flx_flag_;
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_, cwidth_;

};

#endif // CRINTEGRATORS_HPP
