#include "bits/stdc++.h"
using namespace std;
using ll = long long;
using lll = __int128;
#define all(x) (x).begin(),(x).end()

template<class T>
struct min25
//G 需要提供:
//vector<T> c; 常数系数，不能依赖 x 或 p
//vector<T> pre(ll x); pre(x)[i]=sum_{1<=j<=x} g_i(j)
//T f(ll p, int e, ll pe); 返回目标函数在 p^e 处的值
//要求每个 g_i 完全积性，且对任意素数 p 有 f(p)=sum c[i]*g_i(p)
//合数和素数幂由 f(p, e, pe) 负责，不要求上式对合数成立
//符号和倍数写进 c，不要写进 pre 对应的基函数
{
	ll n, sq;
	vector<ll> w;
	vector<T> ps, sps;//ps[getid(x)]=sum_{p<=x} f(p)，sps[i]=sum_{j<i} f(pr[j])
	vector<int> id1, id2, pr, lim;//lim[id]: p_i^2<=w[id] 的个数
	vector<vector<T>> mem;//只记 s<lim[id] 的状态
	vector<vector<char>> flg;
	T ans;

	min25(ll _n): n(_n)
	{
		sq = isqrt(n);
		id1.assign(sq + 2, -1);
		id2.assign(sq + 2, -1);
		for (ll l = 1, r; l <= n; l = r + 1)
		{
			ll x = n / l;
			r = n / x;
			int id = w.size();
			w.push_back(x);
			if (x <= sq) id1[x] = id;
			else id2[n / x] = id;
		}
		sieve(sq);
		int m = w.size(), c = pr.size();
		lim.assign(m, 0);
		for (int id = 0; id < m; id++)
			while (lim[id] < c && lll(pr[lim[id]]) * pr[lim[id]] <= w[id])
				lim[id]++;
	}
	static ll isqrt(ll x)
	{
		ll r = sqrtl(1.0L * x);
		while (lll(r + 1) * (r + 1) <= x) r++;
		while (lll(r) * r > x) r--;
		return r;
	}
	int getid(ll x) const
	{
		if (x <= sq) return id1[x];
		return id2[n / x];
	}
	void sieve(int n)
	{
		vector<int> lp(n + 1);
		pr.clear();
		int i;
		for (i = 2; i <= n; i++)
		{
			if (!lp[i])
			{
				lp[i] = i;
				pr.push_back(i);
			}
			for (int p : pr)
			{
				if (p > lp[i] || p * i > n) break;
				lp[p * i] = p;
			}
		}
	}
	template<class G>
	void build(G &g)
	{
		ps.clear(); sps.clear();
		vector<T> c = g.c, g1 = g.pre(1);
		int k = c.size(), g1n = g1.size();
		assert(g1n == k);
		vector<vector<T>> h(k);
		int m = w.size();
		for (ll x : w)
		{
			vector<T> gx = g.pre(x);
			int gxn = gx.size();
			assert(gxn == k);
			for (int j = 0; j < k; j++) h[j].push_back(gx[j] - g1[j]);
		}
		vector<T> s(k);
		sps.push_back(0);
		for (int p : pr)
		{
			vector<T> gp = g.pre(p), gl = g.pre(p - 1);
			vector<T> old = s;
			int gpn = gp.size(), gln = gl.size();
			assert(gpn == k);
			assert(gln == k);
			for (int j = 0; j < k; j++) gp[j] -= gl[j];
			ll pp = 1LL * p * p;
			for (int id = 0; id < m && w[id] >= pp; id++)
			{
				ll x = w[id];
				int to = getid(x / p);
				for (int j = 0; j < k; j++) h[j][id] -= gp[j] * (h[j][to] - old[j]);
			}
			T sum = 0;
			for (int j = 0; j < k; j++)
			{
				s[j] += gp[j];
				sum += c[j] * s[j];
			}
			sps.push_back(sum);
		}
		ps.assign(w.size(), 0);
		for (int j = 0; j < k; j++)
			for (int id = 0; id < m; id++)
				ps[id] += c[j] * h[j][id];
		mem.assign(m, { });
		flg.assign(m, { });
		for (int id = 0; id < m; id++)
		{
			mem[id].resize(lim[id]);
			flg[id].assign(lim[id], 0);
		}
	}
	template<class G>
	T dfs(ll x, int s, G &g)
	{
		if (x < 2) return T(0);
		int xid = getid(x);
		if (s < lim[xid] && flg[xid][s]) return mem[xid][s];
		T res = ps[xid] - sps[s];
		for (int i = s; i < lim[xid]; i++)
		{
			ll p = pr[i];
			ll pe = p;
			for (int e = 1; lll(pe) * p <= x; e++)
			{
				ll pe1 = pe * p;
				res += g.f(p, e, pe) * dfs(x / pe, i + 1, g) + g.f(p, e + 1, pe1);
				pe = pe1;
			}
		}
		if (s < lim[xid])
		{
			flg[xid][s] = 1;
			mem[xid][s] = res;
		}
		return res;
	}
	template<class G>
	T solve(G g, T one = T(1))
	{
		build(g);
		return ans = one + dfs(n, 0, g);
	}
};

namespace min25_example
{
const int mod = 469762049;

struct Z
{
	int x;

	Z(ll v = 0)
	{
		v %= mod;
		if (v < 0) v += mod;
		x = v;
	}
	Z operator+(const Z &b) const { return x + b.x; }
	Z operator-(const Z &b) const { return x - b.x; }
	Z operator*(const Z &b) const { return 1LL * x * b.x; }
	Z &operator+=(const Z &b) { return *this = *this + b; }
	Z &operator-=(const Z &b) { return *this = *this - b; }
	Z &operator*=(const Z &b) { return *this = *this * b; }
	bool operator==(const Z &b) const { return x == b.x; }
};

ostream &operator<<(ostream &cout, const Z &a)
{
	return cout << a.x;
}

struct ae_bp
{
	Z a, b;
	vector<Z> c;
	ae_bp(ll _a, ll _b): a(_a), b(_b), c({_a, _b}) { }
	vector<Z> pre(ll x) const { return {x, Z(x) * Z(x + 1) * Z((mod + 1) / 2)}; }//f(p)=a+b*p
	Z f(ll p, int e, ll) const { return a * e + b * p; }
};
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int t;
	cin >> t;
	while (t--)
	{
		ll n, a, b;
		cin >> n >> a >> b;
		cout << min25<min25_example::Z>(n).solve(min25_example::ae_bp(a, b)) << '\n';
	}
	return 0;
}
