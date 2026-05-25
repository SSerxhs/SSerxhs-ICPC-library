namespace cipolla
{
	typedef unsigned int ui;
	typedef unsigned long long ull;
	ui p, w;
	struct Q
	{
		ull x, y;
		Q operator*(const Q &o) const { return {(x * o.x + y * o.y % p * w) % p, (x * o.y + y * o.x) % p}; }
	};
	ui ksm(ull x, ui y)
	{
		ull r = 1;
		while (y)
		{
			if (y & 1) r = r * x % p;
			x = x * x % p; y >>= 1;
		}
		return r;
	}
	Q ksm(Q x, ui y)
	{
		Q r = {1, 0};
		while (y)
		{
			if (y & 1) r = r * x;
			x = x * x; y >>= 1;
		}
		return r;
	}
	ui mosqrt(ui x, ui P)//P 为素数，0<=x<P
	{
		if (x == 0 || P == 2) return x;
		p = P;
		if (ksm(x, p - 1 >> 1) != 1) return -1;
		ui y;
		mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
		do y = rnd() % p, w = ((ull)y * y + p - x) % p; while (ksm(w, p - 1 >> 1) <= 1);//not for p=2
		y = ksm({y, 1}, p + 1 >> 1).x;
		if (y * 2 > p) y = p - y;//两解取小
		return y;
	}
}
using cipolla::mosqrt;

