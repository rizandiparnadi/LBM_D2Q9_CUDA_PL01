#pragma once

#include <vector>


// One-dimensional array class
template<typename T>
class Array1D
{
public:
	Array1D() :
		m_Width(0)
	{};

	Array1D(size_t x, T initValue = 0){
		init(x, initValue);
	};

	void init(size_t x, T initValue = 0) {
		m_Width = x;
		m_Data.resize(x, initValue);
	};

	inline T& operator()(size_t x) {
		return m_Data.at(x);
	};

	inline T* data()	{ return &m_Data[0]; };
	inline size_t width()	const	{ return m_Width; };
	inline size_t stride()	const	{ return sizeof(T); };
	inline size_t size()	const	{ return m_Width*sizeof(T); };
	inline size_t elements() const	{ return m_Width; };

private:
	size_t m_Width;
	std::vector<T> m_Data;
};

typedef Array1D<float> FloatArray1D;
typedef Array1D<double> DoubleArray1D;
typedef Array1D<int> IntArray1D;
typedef Array1D<unsigned int> UintArray1D;
