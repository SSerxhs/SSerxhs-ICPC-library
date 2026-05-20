
namespace sh
{
	using ull = unsigned long long;
	using lll = __uint128_t;
	const int N = 1e6 + 5;
	const ull p = (ull)1e18 - 11, b = 137;
	ull m[N];
	int i = []() {
		m[0] = 1;
		for (int i = 1; i < N; i++) m[i] = (lll)m[i - 1] * b % p;
		return 0;
	}();
	struct str
	{
		int n;
		vector<ull> a;
		template<class T> str(const vector<T> &s) :n(s.size()), a(n + 1)
		{
			for (i = 0; i < n; i++) a[i + 1] = ((lll)a[i] * b + s[i]) % p;
		}
		template<class T> str(const basic_string<T> &s) : n(s.size()), a(n + 1)//直接去掉模板换成 string 也可以
		{
			for (i = 0; i < n; i++) a[i + 1] = ((lll)a[i] * b + s[i]) % p;
		}
		ull getv(int l, int r)//[l,r)
		{
			return (a[r] + (lll)(p - a[l]) * m[r - l]) % p;
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

