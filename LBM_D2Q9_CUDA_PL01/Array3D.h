#pragma once

#include <vector>


// Three-dimensional array class
template<typename T>
class Array3D
{
public:
	Array3D() :
	  m_Width(0), m_Height(0), m_Depth(0)
	  {};

    Array3D(size_t x, size_t y, size_t z, T initValue = 0){
		init(x, y, z, initValue);
	};

	void init(size_t x, size_t y, size_t z, T initValue = 0) {
		m_Width = x;
		m_Height = y;
		m_Depth = z;
		m_Data.resize(x*y*z, initValue);
	};

    inline T& operator()(size_t x, size_t y, size_t z) {
        return m_Data.at(x + y * m_Width + z * m_Width * m_Height);
    };

	inline T* data()	{ return &m_Data[0]; };
	inline size_t width()	const	{ return m_Width; };
	inline size_t height()	const	{ return m_Height; };
	inline size_t depth()	const	{ return m_Depth; };
	inline size_t stride()	const	{ return sizeof(T); };
	inline size_t pitch()	const	{ return m_Width*sizeof(T); };
	inline size_t slicePitch()	const	{ return m_Width*m_Depth*sizeof(T); };
	inline size_t size()	const	{ return m_Width*m_Height*m_Depth*sizeof(T); };
	inline size_t elements() const	{ return m_Width*m_Height*m_Depth; };

private:
	size_t m_Width, m_Height, m_Depth;
	std::vector<T> m_Data;
};

typedef Array3D<float> FloatArray3D;
typedef Array3D<double> DoubleArray3D;
typedef Array3D<int> IntArray3D;
typedef Array3D<unsigned int> UintArray3D;
