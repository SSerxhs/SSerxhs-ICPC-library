void fwt_and(vector<ull> &A)//本质：母集和
{
	ull n = A.size(), *a = A.data(), i, j, k, l, *f, *g;
	for (i = 1; i < n; i = l)
	{
		l = i * 2;
		for (j = 0; j < n; j += l)
		{
			f = a + j; g = a + j + i;
			for (k = 0; k < i; k++) f[k] += g[k];
		}
		if (l == n || i == 1 << 10) for (ull &x : A) x %= p;
	}
}
void ifwt_and(vector<ull> &A)
{
	ull n = A.size(), *a = A.data(), i, j, k, l, *f, *g;
	for (i = 1; i < n; i = l)
	{
		l = i * 2;
		for (j = 0; j < n; j += l)
		{
			f = a + j; g = a + j + i;
			for (k = 0; k < i; k++) f[k] += p * i - g[k];
		}
		if (l == n || i == 1 << 10) for (ull &x : A) x %= p;
	}
}
void fwt_or(vector<ull> &A)//本质：子集和
{
	ull n = A.size(), *a = A.data(), i, j, k, l, *f, *g;
	for (i = 1; i < n; i = l)
	{
		l = i * 2;
		for (j = 0; j < n; j += l)
		{
			f = a + j; g = a + j + i;
			for (k = 0; k < i; k++) g[k] += f[k];
		}
		if (l == n || i == 1 << 10) for (ull &x : A) x %= p;
	}
}
void ifwt_or(vector<ull> &A)
{
	ull n = A.size(), *a = A.data(), i, j, k, l, *f, *g;
	for (i = 1; i < n; i = l)
	{
		l = i * 2;
		for (j = 0; j < n; j += l)
		{
			f = a + j; g = a + j + i;
			for (k = 0; k < i; k++) g[k] += p * i - f[k];
		}
		if (l == n || i == 1 << 10) for (ull &x : A) x %= p;
	}
}
void fwt_xor(vector<ull> &A)
{
	ull n = A.size(), *a = A.data(), i, j, k, l, *f, *g;
	for (i = 1; i < n; i = l)
	{
		l = i * 2;
		for (j = 0; j < n; j += l)
		{
			f = a + j; g = a + j + i;
			for (k = 0; k < i; k++)
			{
				if ((f[k] += g[k]) >= p) f[k] -= p;
				g[k] = (f[k] + 2 * (p - g[k])) % p;
			}
		}
	}
}
void ifwt_xor(vector<ull> &A)
{
	ull n = A.size(), *a = A.data(), i, j, k, l, *f, *g, x = p + 1 >> 1, y = 1;
	for (i = 1; i < n; i = l)
	{
		l = i * 2;
		for (j = 0; j < n; j += l)
		{
			f = a + j; g = a + j + i;
			for (k = 0; k < i; k++)
			{
				if ((f[k] += g[k]) >= p) f[k] -= p;
				g[k] = (f[k] + 2 * (p - g[k])) % p;
			}
		}
		y = y * x % p;
	}
	for (i = 0; i < n; i++) a[i] = a[i] * y % p;
}
vector<ull> fst(const vector<ull> &s, const vector<ull> &t)
{
	int n = s.size(), m = __builtin_ctz(n), i, j, k;
	vector<ull> a[m + 1], b[m + 1], c[m + 1], r(n);
	for (i = 0; i <= m; i++) a[i].resize(n), b[i].resize(n), c[i].resize(n);
	for (i = 0; i < n; i++)
	{
		k = __builtin_popcount(i);
		a[k][i] = s[i];
		b[k][i] = t[i];
	}
	for (i = 0; i < m; i++) fwt_or(a[i]), fwt_or(b[i]);//如果魔改，上限需改为 m
	for (i = 0; i <= m; i++) for (j = 0; j <= i; j++) for (k = 0; k < n; k++) c[i][k] = (c[i][k] + (ull)a[j][k] * b[i - j][k]) % p;
	for (i = 1; i <= m; i++) ifwt_or(c[i]);//如果魔改，下限需改为 0
	for (i = 0; i < n; i++) r[i] = c[__builtin_popcount(i)][i];
	return r;
}

