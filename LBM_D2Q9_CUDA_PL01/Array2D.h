#pragma once

#include <vector>


// Two-dimensional array class
template<typename T>
class Array2D
{
public:
	Array2D() :
		m_Width(0), m_Height(0)
	{};

	Array2D(size_t x, size_t y, T initValue = 0){
		init(x, y, initValue);
	};

	void init(size_t x, size_t y, T initValue = 0) {
		m_Width = x;
		m_Height = y;
		m_Data.resize(x*y, initValue);
	};

	inline T& operator()(size_t x, size_t y) {
		return m_Data.at(x + y * m_Width);
	};

	inline T* data()	{ return &m_Data[0]; };
	inline size_t width()	const	{ return m_Width; };
	inline size_t height()	const	{ return m_Height; };
	inline size_t stride()	const	{ return sizeof(T); };
	inline size_t pitch()	const	{ return m_Width*sizeof(T); };
	inline size_t size()	const	{ return m_Width*m_Height*sizeof(T); };
	inline size_t elements() const	{ return m_Width*m_Height; };

private:
	size_t m_Width, m_Height;
	std::vector<T> m_Data;
};

typedef Array2D<float> FloatArray2D;
typedef Array2D<double> DoubleArray2D;
typedef Array2D<int> IntArray2D;
typedef Array2D<unsigned int> UintArray2D;
