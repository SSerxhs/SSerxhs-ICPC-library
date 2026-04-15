namespace online_inv
{
	typedef unsigned int ui;
	typedef unsigned long long ull;
	const ull p = 1e9 + 7, n = 1010, m = n * n, N = m + 2;
	static_assert(n *n *n > p);
	int l[N], r[N];
	ull y[N];
	bool s[N];
	ull _inv[N * 2], i, j, k;
	void init_inv()
	{
		_inv[1] = 1;
		for (i = 2; i < m * 2; i++)
		{
			j = p / i;
			_inv[i] = (p - j) * _inv[p - i * j] % p;
		}
		s[0] = y[0] = 1;
		for (i = 1; i < n; i++) for (j = i; j < n; j++) if (!s[k = i * m / j])
		{
			y[k] = j;
			s[k] = 1;
		}
		l[0] = 1;
		for (i = 1; i <= m; i++) l[i] = s[i] ? y[i] : l[i - 1];
		r[m] = 1;
		for (i = m - 1; ~i; i--) r[i] = s[i] ? y[i] : r[i + 1];
		for (i = 0; i <= m; i++) y[i] = min(l[i], r[i]);
	}
	inline ull inv(const ull &x)
	{
		assert(x && x < p);
		if (x < m * 2) return _inv[x];
		k = x * m / p;
		j = y[k] * x % p;
		return (j < m * 2 ? _inv[j] : p - _inv[p - j]) * y[k] % p;
	}
	bool _ = (init_inv(), 0);
}
using online_inv::inv, online_inv::p;

