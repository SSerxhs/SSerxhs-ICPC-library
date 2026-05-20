namespace sh
{
	using ui = unsigned int;
	using ull = unsigned long long;
	const int N = 1e6 + 5;
	const ui p1 = 2'034'452'107, p2 = 2'013'074'419;
	struct pa
	{
		ui v1, v2;
		pa(ui v = 0) :v1(v), v2(v) { }
		pa(ui v1, ui v2) :v1(v1), v2(v2) { }
		pa operator*(const pa &o) const { return pa(1llu * v1 * o.v1 % p1, 1llu * v2 * o.v2 % p2); }
	};
	pa fma(const pa &a, const pa &b, const pa &c) { return pa((1llu * a.v1 * b.v1 + c.v1) % p1, (1llu * a.v2 * b.v2 + c.v2) % p2); }
	const pa b = {137, 149}, inv = {1'603'801'661, 1'024'053'074};
	pa m[N];
	int i = []() {
		m[0] = {p1 - 1, p2 - 1};
		for (int i = 1; i < N; i++) m[i] = m[i - 1] * b;
		return 0;
	}();
	struct str
	{
		int n;
		vector<pa> a;
		template<class T> str(const vector<T> &s) :n(s.size()), a(n + 1)
		{
			for (i = 0; i < n; i++) a[i + 1] = fma(a[i], b, s[i]);
		}
		template<class T> str(const basic_string<T> &s) : n(s.size()), a(n + 1)//直接去掉模板换成 string 也可以
		{
			for (i = 0; i < n; i++) a[i + 1] = fma(a[i], b, s[i]);
		}
		ull getv(int l, int r)//[l,r)
		{
			auto [x, y] = fma(a[l], m[r - l], a[r]);
			return (ull)x << 32 | y;
		}
		int lcp(int i, int j)
		{
			if (i == j) return n - i;
			int l = 0, r = n - max(i, j), mid;
			while (l < r)
			{
				mid = (l + r + 1) >> 1;
				if (getv(i, i + mid) == getv(j, j + mid)) l = mid;
				else r = mid - 1;
			}
			return l;
		}
	};
}
using sh::str;
