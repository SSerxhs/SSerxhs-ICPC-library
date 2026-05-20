template<class T> struct bit
{
	vector<T> a;
	int n;
	bit() { }
	bit(int nn) :n(nn), a(nn + 1) { }
	template<class TT> bit(int nn, TT *b) : n(nn), a(nn + 1)
	{
		for (int i = 1; i <= n; i++) a[i] = b[i];
		for (int i = 1; i <= n; i++) if (i + (i & -i) <= n) a[i + (i & -i)] += a[i];
	}
	void add(int x, T y)
	{
		//cerr<<"add "<<x<<" by "<<y<<endl;
		assert(1 <= x && x <= n);
		a[x] += y;
		while ((x += x & -x) <= n) a[x] += y;
	}
	T sum(int x)
	{
		//cerr<<"sum "<<x;
		assert(0 <= x && x <= n);
		T r = a[x];
		while (x ^= x & -x) r += a[x];
		//cerr<<"= "<<r<<endl;
		return r;
	}
	T sum(int x, int y)
	{
		return sum(y) - sum(x - 1);
	}
	int lower_bound(T x)
	{
		if (n == 0 || x <= 0) return 0;
		int i = __lg(n), j = 0;
		for (; i >= 0; i--) if ((1 << i | j) <= n && a[1 << i | j] < x) j |= 1 << i, x -= a[j];
		return j + 1;
	}
};
