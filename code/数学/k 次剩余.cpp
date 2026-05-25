
namespace get_root
{
	bool ied = 0;
	const int N = 1e5 + 5;
	vector<ui> pr;
	bool ed[N];
	void init()
	{
		pr.reserve(N);
		for (ui i = 2; i < N; i++)
		{
			if (!ed[i]) pr.push_back(i);
			for (ui x : pr)
			{
				if (i * x >= N) break;
				ed[i * x] = 1;
				if (i % x == 0) break;
			}
		}
	}
	ui ksm(ui x, ui y, ui p)
	{
		ui r = 1;
		while (y)
		{
			if (y & 1) r = (ull)r * x % p;
			x = (ull)x * x % p; y >>= 1;
		}
		return r;
	}
	vector<ui> getw(ui n)
	{
		vector<ui> w;
		for (ui x : pr)
		{
			if (x * x > n) break;
			if (n % x == 0)
			{
				w.push_back(x);
				n /= x;
				for (ui i = n / x; n == x * i; i = n / x) n /= x;
			}
		}
		if (n > 1) w.push_back(n);
		return w;
	}
	int getrt(ui n)
	{
		if (n <= 2) return n - 1;
		if (!ed[4]) init();
		auto w = getw(n);
		ui ph = n;
		for (ui x : w) ph = ph / x * (x - 1);
		w = getw(ph);
		for (ui &x : w) x = ph / x;
		for (ui i = 2; i < n; i++) if (gcd(i, n) == 1)
		{
			for (ui x : w) if (ksm(i, x, n) == 1) goto no;
			return i;
		no:;
		}
		return -1;
	}
}
namespace BSGS
{
	using ui = unsigned;
	using ull = unsigned long long;
	template<int N, class T, class TT> struct ht//个数，定义域，值域
	{
		const static int p = 1e6 + 7, M = p + 2;
		TT a[N];
		T v[N];
		int fir[p + 2], nxt[N], st[p + 2];//和模数相适应
		int tp, ds;//自定义模数
		ht() { memset(fir, 0, sizeof fir); tp = ds = 0; }
		void mdf(T x, TT z)//位置，值
		{
			ui y = x % p;
			for (int i = fir[y]; i; i = nxt[i]) if (v[i] == x) return a[i] = z, void();//若不可能重复不需要 for
			v[++ds] = x; a[ds] = z;
			if (!fir[y]) st[++tp] = y;
			nxt[ds] = fir[y]; fir[y] = ds;
		}
		TT find(T x)
		{
			ui y = x % p;
			int i;
			for (i = fir[y]; i; i = nxt[i]) if (v[i] == x) return a[i];
			return 0;//返回值和是否判断依据要求决定
		}
		void clear()
		{
			++tp;
			while (--tp) fir[st[tp]] = 0;
			ds = 0;
		}
	};
	const int N = 5e4;
	ht<N, ui, ui> s;
	int exgcd(int a, int b)
	{
		if (a == 1) return 1;
		return (1 - (ll)b * exgcd(b % a, a)) / a;//not ull
	}
	int bsgs(ui a, ui b, ui p)
	{
		s.clear();
		a %= p; b %= p;
		if (!a) return 1 - min((int)b, 2);//含 -1
		ui i, j, k, x, y;
		x = sqrt(p) + 2;
		for (i = 0, j = 1; i < x; i++, j = (ull)j * a % p)
		{
			if (j == b) return i;
			s.mdf((ull)j * b % p, i + 1);
		}
		k = j;
		for (i = 1; i <= x; i++, j = (ull)j * k % p) if (y = s.find(j)) return (ull)i * x - y + 1;
		return -1;
	}
	bool isprime(ui p)
	{
		if (p <= 1) return 0;
		for (ui i = 2; i * i <= p; i++) if (p % i == 0) return 0;
		return 1;
	}
	int exbsgs(ui a, ui b, ui p)//a^x=b(mod p)
	{
		//if (isprime(p)) return bsgs(a,b,p);
		a %= p; b %= p;
		ui i, j, k, x, y = __lg(p), cnt = 0;
		for (i = 0, j = 1 % p; i <= y; i++, j = (ull)j * a % p) if (j == b) return i;
		y = 1;
		while (1)
		{
			if ((x = gcd(a, p)) == 1) break;
			if (b % x) return -1;//no sol
			++cnt;
			p /= x; b /= x;
			y = (ull)y * (a / x) % p;
		}
		a %= p;
		b = (ull)b * (p + exgcd(y, p)) % p;
		int r = bsgs(a, b, p);
		return r == -1 ? -1 : r + cnt;
	}
}
pair<ll, ll> exgcd(ll a, ll b, ll c)//ax+by=c，{-1,-1} 无解，b=0 返回 {c/a,0}，否则返回最小非负 x
{
	assert(a || b);
	if (!b) return {c / a, 0};
	if (a < 0) a = -a, b = -b, c = -c;
	ll d = gcd(a, b);
	if (c % d) return {-1, -1};
	ll x = 1, x1 = 0, p = a, q = b, k;
	b = abs(b);
	while (b)
	{
		k = a / b;
		x -= k * x1; a -= k * b;
		swap(x, x1);
		swap(a, b);
	}
	b = abs(q / d);
	x = x * (c / d) % b;
	if (x < 0) x += b;
	return {x, (c - p * x) / q};
}
ll fun(ll a, ll b, ll p)//ax=b(mod p)
{
	return exgcd(-p, a, b).second % p;
}
using get_root::getrt;
using BSGS::bsgs, BSGS::exbsgs;
int nth_root(ui k, ui y, ui p)//x^k=y(mod p)
{
	if (k == 0) return y == 1 ? 0 : -1;
	if (y == 0) return 0;
	int g = getrt(p);
	if (g == -1) return -1;
	ll z = bsgs(g, y, p);
	if (z == -1) return -1;
	ll x = fun(k, z, p - 1);
	if (x == -1) return -1;
	return get_root::ksm(g, x, p);
}
