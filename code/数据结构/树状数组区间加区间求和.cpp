
template<class T> struct bit
{
	vector<T> a, b;
	int n;
	template<class TT> bit(int n, TT *c) :n(n), a(n + 1), b(n + 1)
	{
		for (int i = 1; i <= n; i++) b[i] = -c[i];
		for (int i = 1; i <= n; i++) if (i + (i & -i) <= n) b[i + (i & -i)] += b[i];
	}
	void add(int l, int r, T d)
	{
		T x;
		int i;
		for (i = l, x = d * i; i <= n; i += i & -i) a[i] += d, b[i] += x;
		for (i = r + 1, x = d * i; i <= n; i += i & -i) a[i] -= d, b[i] -= x;
	}
	void add(int x, T d)
	{
		for (int i = x; i <= n; i += i & -i) b[i] -= d;
	}
	T sum(int x)
	{
		T r1 = 0, r2 = 0;
		for (int i = x; i; i ^= i & -i) r1 += a[i], r2 += b[i];
		return r1 * (x + 1) - r2;
	}
	T sum(int l, int r)
	{
		return sum(r) - sum(l - 1);
	}
};
