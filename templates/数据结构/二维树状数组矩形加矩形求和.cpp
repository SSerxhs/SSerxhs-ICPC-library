template <class T> struct bit
{
	int n, m;
	vector<vector<T>> a, b, c, d;
private:
	void modify(vector<vector<T>> &a, int x, int y, T z)
	{
		for (int i = x; i <= n; i += i & -i) for (int j = y; j <= m; j += j & -j) a[i][j] += z;
	}
	T ask(const vector<vector<T>> &a, int x, int y) const
	{
		T res = 0; --x; --y;
		for (int i = x; i; i ^= i & -i) for (int j = y; j; j ^= j & -j) res += a[i][j];
		return res;
	}
	void cg(int x, int y, T t)
	{
		if (x > n || y > n) return;
		modify(a, x, y, t);
		modify(b, x, y, x * t);
		modify(c, x, y, y * t);
		modify(d, x, y, x * y * t);
	}
public:
	bit(int n, int m) :n(n), m(m), a(n + 1, vector<T>(m + 1)), b(a), c(a), d(a) { }
	void add(int x1, int y1, int x2, int y2, T t)
	{
		++x2, ++y2;
		cg(x1, y1, t); cg(x2, y2, t);
		cg(x1, y2, -t); cg(x2, y1, -t);
	}
	T sum(int x, int y) const
	{
		if (x <= 0 || y <= 0) return 0;
		++x; ++y;
		return ask(a, x, y) * x * y + ask(d, x, y) - ask(b, x, y) * y - ask(c, x, y) * x;
	}
	T sum(int x1, int y1, int x2, int y2) const
	{
		--x1; --y1;
		return sum(x2, y2) + sum(x1, y1) - sum(x2, y1) - sum(x1, y2);
	}
};
