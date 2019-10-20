#pragma once

#include "cuda_runtime.h"
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__

__device__ __forceinline__ float2 operator-(float2& a)
{
	return {-a.x, -a.y};
}

__device__ __forceinline__ float2 operator+(float2 a, float2 b)
{
	return {a.x + b.x, a.y + b.y};
}
__device__ __forceinline__ void operator+=(float2& a, float2 b)
{
	a.x += b.x;
	a.y += b.y;
}
__device__ __forceinline__ float2 operator+(float2 a, float b)
{
	return {a.x + b, a.y + b};
}
__device__ __forceinline__ float2 operator+(float b, float2 a)
{
	return {a.x + b, a.y + b};
}
__device__ __forceinline__ void operator+=(float2& a, float b)
{
	a.x += b;
	a.y += b;
}

__device__ __forceinline__ float2 operator-(float2 a, float2 b)
{
	return {a.x - b.x, a.y - b.y};
}
__device__ __forceinline__ void operator-=(float2& a, float2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
__device__ __forceinline__ float2 operator-(float2 a, float b)
{
	return {a.x - b, a.y - b};
}
__device__ __forceinline__ float2 operator-(float b, float2 a)
{
	return {b - a.x, b - a.y};
}
__device__ __forceinline__ float2 operator*(float2 a, float2 b)
{
	return {a.x * b.x, a.y * b.y};
}
__device__ __forceinline__ void operator*=(float2& a, float2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
__device__ __forceinline__ float2 operator*(float2 a, float b)
{
	return {a.x * b, a.y * b};
}
__device__ __forceinline__ float2 operator*(float b, float2 a)
{
	return {b * a.x, b * a.y};
}
__device__ __forceinline__ void operator*=(float2& a, float b)
{
	a.x *= b;
	a.y *= b;
}

__device__ __forceinline__ float2 operator/(float2 a, float2 b)
{
	return {a.x / b.x, a.y / b.y};
}
__device__ __forceinline__ void operator/=(float2& a, float2 b)
{
	a.x /= b.x;
	a.y /= b.y;
}
__device__ __forceinline__ float2 operator/(float2 a, float b)
{
	return {a.x / b, a.y / b};
}
__device__ __forceinline__ void operator/=(float2& a, float b)
{
	a.x /= b;
	a.y /= b;
}
__device__ __forceinline__ float2 operator/(float b, float2 a)
{
	return {b / a.x, b / a.y};
}
__device__ __forceinline__ float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
__device__ __forceinline__ float length(float2 v)
{
	return sqrtf(dot(v, v));
}
__device__ __forceinline__ float2 normalize(float2 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}
__device__ __forceinline__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 operator*(float3 a, float b)
{
	return { a.x * b, a.y * b, a.z * b };
}
__device__ __forceinline__ float3 operator*(float b, float3 a)
{
	return { a.x * b, a.y * b, a.z * b };
}
__device__ __forceinline__ float3 normalize(float3 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

__device__ __forceinline__ double2 operator-(double2& a)
{
	return { -a.x, -a.y };
}

__device__ __forceinline__ double2 operator+(double2 a, double2 b)
{
	return { a.x + b.x, a.y + b.y };
}
__device__ __forceinline__ void operator+=(double2& a, double2 b)
{
	a.x += b.x;
	a.y += b.y;
}
__device__ __forceinline__ double2 operator+(double2 a, double b)
{
	return { a.x + b, a.y + b };
}
__device__ __forceinline__ double2 operator+(double b, double2 a)
{
	return { a.x + b, a.y + b };
}
__device__ __forceinline__ void operator+=(double2& a, double b)
{
	a.x += b;
	a.y += b;
}

__device__ __forceinline__ double2 operator-(double2 a, double2 b)
{
	return { a.x - b.x, a.y - b.y };
}
__device__ __forceinline__ void operator-=(double2& a, double2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
__device__ __forceinline__ double2 operator-(double2 a, double b)
{
	return { a.x - b, a.y - b };
}
__device__ __forceinline__ double2 operator-(double b, double2 a)
{
	return { b - a.x, b - a.y };
}
__device__ __forceinline__ double2 operator*(double2 a, double2 b)
{
	return { a.x * b.x, a.y * b.y };
}
__device__ __forceinline__ void operator*=(double2& a, double2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
__device__ __forceinline__ double2 operator*(double2 a, double b)
{
	return { a.x * b, a.y * b };
}
__device__ __forceinline__ double2 operator*(double b, double2 a)
{
	return { b * a.x, b * a.y };
}
__device__ __forceinline__ void operator*=(double2& a, double b)
{
	a.x *= b;
	a.y *= b;
}

__device__ __forceinline__ double2 operator/(double2 a, double2 b)
{
	return { a.x / b.x, a.y / b.y };
}
__device__ __forceinline__ void operator/=(double2& a, double2 b)
{
	a.x /= b.x;
	a.y /= b.y;
}
__device__ __forceinline__ double2 operator/(double2 a, double b)
{
	return { a.x / b, a.y / b };
}
__device__ __forceinline__ void operator/=(double2& a, double b)
{
	a.x /= b;
	a.y /= b;
}
__device__ __forceinline__ double2 operator/(double b, double2 a)
{
	return { b / a.x, b / a.y };
}
__device__ __forceinline__ double dot(double2 a, double2 b)
{
	return a.x * b.x + a.y * b.y;
}
__device__ __forceinline__ double length(double2 v)
{
	return sqrt(dot(v, v));
}
__device__ __forceinline__ double2 normalize(double2 v)
{
	double invLen = rsqrt(dot(v, v));
	return v * invLen;
}
__device__ __forceinline__ double dot(double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ double3 operator*(double3 a, double b)
{
	return { a.x * b, a.y * b, a.z * b };
}
__device__ __forceinline__ double3 operator*(double b, double3 a)
{
	return { a.x * b, a.y * b, a.z * b };
}
__device__ __forceinline__ double3 normalize(double3 v)
{
	double invLen = rsqrt(dot(v, v));
	return v * invLen;
}
__device__ __forceinline__ void operator/=(double3& a, double b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

__device__ __forceinline__ void operator*=(double3& a, double b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}
