struct nd
{
	ll x, y, sy;
	nd operator+(const nd &o) const
	{
		return {x + o.x, y + o.y, sy + o.sy + y * o.x};
	}
};
nd ksm(nd a, ll k)
{
	nd res{ };
	while (k)
	{
		if (k & 1) res = res + a;
		a = a + a; k >>= 1;
	}
	return res;
}
nd sol(int a, int b, int m, int n, nd dx, nd dy)//[0,n] (ax+b)/m 0<=b<m
{
	if (!n) return { };
	if (a >= m) return sol(a % m, b, m, n, ksm(dy, a / m) + dx, dy);
	ll c = ((ll)n * a + b) / m;
	if (!c) return ksm(dx, n);
	ll cnt = n - ((ll)m * c - b - 1) / a;
	return ksm(dx, (m - b - 1) / a) + dy + sol(m, (m - b - 1) % a, a, c - 1, dy, dx) + ksm(dx, cnt);
}
ll sum_of_floor_of_linear(int a, int b, int m, int n)//[0,n] sum((ax+b)/m)
{
	nd dx = {1, 0, 0}, dy = {0, 1, 0};
	int nb = (b % m + m) % m;
	return sol(a, nb, m, n, dx, dy).sy + (ll)(b - nb) / m * (n + 1);
}
int min_of_mod_of_linear(int a, int b, int p, int n)//[0,n] min((ax+b) mod p)
{
	ll s = sum_of_floor_of_linear(a, b, p, n);
	int l = 0, r = p - 1, mid;
	while (l < r)
	{
		mid = (l + r + 1) / 2;
		if (sum_of_floor_of_linear(a, b - mid, p, n) >= s) l = mid;
		else r = mid - 1;
	}
	return l;
}
