namespace pr
{
	typedef long long ll;
	typedef __int128 lll;
	typedef pair<ll, int> pa;
	ll ksm(ll x, ll y, const ll p)
	{
		ll r = 1;
		while (y)
		{
			if (y & 1) r = (lll)r * x % p;
			x = (lll)x * x % p; y >>= 1;
		}
		return r;
	}
	namespace miller
	{
		const int p[7] = {2, 3, 5, 7, 11, 61, 24251};
		ll s, t;
		bool test(ll n, int p)
		{
			if (p >= n) return 1;
			ll r = ksm(p, t, n), w;
			for (int j = 0; j < s && r != 1; j++)
			{
				w = (lll)r * r % n;
				if (w == 1 && r != n - 1) return 0;
				r = w;
			}
			return r == 1;
		}
		bool prime(ll n)
		{
			if (n < 2 || n == 46'856'248'255'981ll) return 0;
			for (int i = 0; i < 7; ++i) if (n % p[i] == 0) return n == p[i];
			s = __builtin_ctz(n - 1); t = n - 1 >> s;
			for (int i = 0; i < 7; ++i) if (!test(n, p[i])) return 0;
			return 1;
		}
	}
	using miller::prime;
	mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count());
	namespace rho
	{
		void nxt(ll &x, ll &y, ll &p) { x = ((lll)x * x + y) % p; }
		ll find(ll n, ll C)
		{
			ll l, r, d, p = 1;
			l = rnd() % (n - 2) + 2, r = l;
			nxt(r, C, n);
			int cnt = 0;
			while (l ^ r)
			{
				p = (lll)p * llabs(l - r) % n;
				if (!p) return gcd(n, llabs(l - r));
				++cnt;
				if (cnt == 127)
				{
					cnt = 0;
					d = gcd(llabs(l - r), n);
					if (d > 1) return d;
				}
				nxt(l, C, n); nxt(r, C, n); nxt(r, C, n);
			}
			return gcd(n, p);
		}
		vector<pa> w;
		vector<ll> d;
		void dfs(ll n, int cnt)
		{
			if (n == 1) return;
			if (prime(n)) return w.emplace_back(n, cnt), void();
			ll p = n, C = rnd() % (n - 1) + 1;
			while (p == 1 || p == n) p = find(n, C++);
			int r = 1; n /= p;
			while (n % p == 0) n /= p, ++r;
			dfs(p, r * cnt); dfs(n, cnt);
		}
		vector<pa> getw(ll n)
		{
			w = vector<pa>(0); dfs(n, 1);
			if (n == 1) return w;
			sort(w.begin(), w.end());
			int i, j;
			for (i = 1, j = 0; i < w.size(); i++) if (w[i].first == w[j].first) w[j].second += w[i].second; else w[++j] = w[i];
			w.resize(j + 1);
			return w;
		}
		void dfss(int x, ll n)
		{
			if (x == w.size()) return d.push_back(n), void();
			dfss(x + 1, n);
			for (int i = 1; i <= w[x].second; i++) dfss(x + 1, n *= w[x].first);
		}
		vector<ll> getd(ll n)
		{
			getw(n); d = vector<ll>(0); dfss(0, 1);
			sort(d.begin(), d.end());
			return d;
		}
	}
	using rho::getw, rho::getd;
	using miller::prime;
}
using pr::getw, pr::getd, pr::prime;
lll roundiv(lll x, lll y)
{
	return x >= 0 ? (x + y / 2) / y : (x - y / 2) / y;
}
struct G
{
	lll x, y;
	G operator~() const { return {x, -y}; }
	lll len2() const { return x * x + y * y; }
	G operator+(const G &o) const { return {x + o.x, y + o.y}; }
	G operator-(const G &o) const { return {x - o.x, y - o.y}; }
	G operator*(const G &o) const { return {x * o.x - y * o.y, x * o.y + y * o.x}; }
	G operator/(const G &o) const
	{
		G t = *this * ~o;
		lll l = o.len2();
		return {roundiv(t.x, l), roundiv(t.y, l)};
	}
	G operator%(const G &o) const { return *this - *this / o * o; }
};
G gcd(G a, G b)
{
	if (a.len2() > b.len2()) swap(a, b);
	while (a.len2())
	{
		b = b % a;
		swap(a, b);
	}
	return b;
}
namespace cipolla
{
	typedef unsigned long long ull;
	typedef __uint128_t ll;
	ull p, w;
	struct Q
	{
		ulll x, y;
		Q operator*(const Q &o) const { return {(x * o.x + y * o.y % p * w) % p, (x * o.y + y * o.x) % p}; }
	};
	ull ksm(ulll x, ull y)
	{
		ulll r = 1;
		while (y)
		{
			if (y & 1) r = r * x % p;
			x = x * x % p; y >>= 1;
		}
		return r;
	}
	Q ksm(Q x, ull y)
	{
		Q r = {1, 0};
		while (y)
		{
			if (y & 1) r = r * x;
			x = x * x; y >>= 1;
		}
		return r;
	}
	ull mosqrt(ull x, ull P)//0<=x<P
	{
		if (x == 0 || P == 2) return x;
		p = P;
		if (ksm(x, p - 1 >> 1) != 1) return -1;
		ull y;
		mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count());
		do y = rnd() % p, w = ((ulll)y * y + p - x) % p; while (ksm(w, p - 1 >> 1) <= 1);//not for p=2
		y = ksm({y, 1}, p + 1 >> 1).x;
		if (y * 2 > p) y = p - y;//两解取小
		return y;
	}
}
using cipolla::mosqrt;
vector<pair<ll, ll>> two_sqr_sum(ll n)//只会返回非负解，按照字典序排序
{
	if (n < 0) return { };
	if (n == 0) return {{0, 0}};
	ll m = __lg(n & -n), d = 1 << m / 2, i;
	n >>= m;
	auto w = getw(n);
	vector<G> r((m & 1) ? vector{G{1, 1}} : vector{G{0, 1}, G{1, 0}});
	for (auto [p, k] : w) if (p % 4 == 1)
	{
		vector<G> pw(k + 1);
		pw[0] = {1, 0};
		pw[1] = gcd(G(p, 0), G(mosqrt(p - 1, p), 1));
		assert(pw[1].len2() == p);
		for (i = 2; i <= k; i++) pw[i] = pw[i - 1] * pw[1];
		vector<G> rr; rr.reserve(r.size() * (k + 1));
		for (i = 0; i <= k; i++)
		{
			G x = pw[i] * ~pw[k - i];
			for (G y : r) rr.push_back(x * y);
		}
		swap(r, rr);
	}
	else
	{
		if (k % 2) return { };
		k /= 2;
		while (k--) d *= p;
	}
	vector<pair<ll, ll>> ans;
	ans.reserve(r.size());
	for (auto [x, y] : r) ans.push_back({abs((ll)x * d), abs((ll)y * d)});
	sort(all(ans));
	ans.resize(unique(all(ans)) - ans.begin());
	return ans;
}

