#pragma once

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Random.h"

template<typename Activation>
class FullyConnected :public Layer {

private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	Matrix m_weight;
	Vector m_bias;
	Matrix m_dw;
	Vector m_db;
	Matrix m_z; // 가중치 결과 값
	Matrix m_a; // 최적화 이후 값
	Matrix m_din;

public:
	FullyConnected(const int in_size, const int out_size) :
		Layer(in_size, out_size)
	{}

	// Layer.h에서 만듬
	// mu는 μ로 표준편차를 의미, sigma는 σ으로 평균을 의미 (normal distribution)
	void init(const Scalar& mu, const Scalar& sigma, RNG& rng) { // RNG는 난수 생성기
		m_weight.resize(this->m_in_size, this->m_out_size); // weight matrix 크기 재조정
		m_bias.resize(this->m_out_size); // bias vector 크기 재조정
		m_dw.resize(this->m_in_size, this->m_out_size);
		m_db.resize(this->m_out_size);

		// 함수 초기화
		internal::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
		internal::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);
	}

	void forward(const Matrix& prev_layer_data) {
		const int nobs = prev_layer_data.col();

		// z = w` * in + b
		m_z.resize(this->m_out_size, nobs); // m_z는 W*x + b의 결과값을 저장하는 공간
		m_z.noalias() = m_weight.transpose() * prev_layer_data; // 입력크기와 출력크기로 설정한 가중치행렬을 회전하여 곱
		m_z.colwise() += bias; // bias 더해주기

		// Apply activation funcation
		m_a.resize(this->m_out_size, nobs);
		Activation::activate(m_z, m_a);
	}

	const Matrix& output() const {
		return m_a;
	}

	void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) {
		// TODO
	}

	const Matrix& backprop_data() const {
		return m_din;
	}
	// 위의 내용이 미분을 적용
	// 아래의 내용이 업데이트 적용
	void update(Optimizer& opt) {
		ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
		ConstAlignedMapVec db(m_db.data(), m_db.size());
		AlignedMapVec w(m_weight.data(), m_weight.size());
		AlignedMapVec b(m_bias.data(), m_bias.size());

		opt.update(dw, w); // 가중치를 미분하고 다음은 가중치 -> 최적화를 통해 update과정
		opt.update(db, b);
	}

	std::vector<Scalar> get_parameter() const {}
	void set_parameters(const std::vector<Scalar>& param) {}
	std::vector<Scalar> get_derivatives() const {}
};