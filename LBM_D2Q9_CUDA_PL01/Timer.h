#pragma once

#include <Windows.h>

// (Kind of) High-performance timer. Windows only atm.
// Rizandi 2010
class Timer
{
public:
	Timer()		{ QueryPerformanceFrequency( &m_frequency ); start(); };

	__forceinline void start();

	__forceinline double getSeconds();
	__forceinline double getMiliseconds();
	__forceinline double getMicroseconds();
	
private:
	LARGE_INTEGER m_frequency, m_counterStart, temp;
};

__forceinline void Timer::start()
{
	QueryPerformanceCounter( &m_counterStart );
}

__forceinline double Timer::getSeconds()
{
	QueryPerformanceCounter( &temp );
	return (temp.QuadPart - m_counterStart.QuadPart)/(double)m_frequency.QuadPart;
}

__forceinline double Timer::getMiliseconds()
{
	QueryPerformanceCounter( &temp );
	return ((temp.QuadPart - m_counterStart.QuadPart)/(double)m_frequency.QuadPart)*1000.0f;
}

__forceinline double Timer::getMicroseconds()
{
	QueryPerformanceCounter( &temp );
	return ((temp.QuadPart - m_counterStart.QuadPart)/(double)m_frequency.QuadPart)*1000000.0f;
}