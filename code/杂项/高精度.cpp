struct bigint;
int cmp(const bigint &a, const bigint &b);
struct bigint
{
	using ull = unsigned long long;
	using lll = unsigned __int128;
	const static ull sign = 1llu << 63;
	const static lll p = 4'179'340'454'199'820'289;
	const static lll g = 3;
	const static ull base = 1e6;
	const static int output_base = 10;
	const static int length = round(log(bigint::base) / log(output_base));
	static_assert(output_base == 10 || output_base == 16, "output_base must be 10 or 16");
	static_assert(round(pow(output_base, length)) == base);
	const static int N = 1 << 23;
	static int r[N];
	static lll w[N];
	bool neg;
	vector<ull> a;
private:
	static lll ksm(lll x, ull y)
	{
		lll r = 1;
		while (y)
		{
			if (y & 1) r = r * x % p;
			x = x * x % p; y >>= 1;
		}
		return r;
	}
	static void init(int n)
	{
		static int pr = 0, pw = 0;
		if (pr == n) return;
		int b = __lg(n) - 1, i, j, k;
		for (i = 1; i < n; i++) r[i] = r[i >> 1] >> 1 | (i & 1) << b;
		if (pw < n)
		{
			for (j = 1; j < n; j = k)
			{
				k = j * 2;
				ull wn = ksm(g, (p - 1) / k);
				w[j] = 1;
				for (i = j + 1; i < k; i++) w[i] = w[i - 1] * wn % p;
			}
			pw = n;
		}
		pr = n;
	}
	static void dft(vector<lll> &a, int o = 0)
	{
		int n = a.size(), i, j, k;
		lll y, *f, *g, *wn, *A = a.data();
		init(n);
		for (i = 1; i < n; i++) if (i < r[i]) swap(A[i], A[r[i]]);
		static const int T = 12;
		static_assert(T + 2 <= numeric_limits<lll>::max() / (p * p));
		for (k = 1; k < n; k *= 2)
		{
			wn = w + k;
			for (i = 0; i < n; i += k * 2)
			{
				f = A + i; g = A + i + k;
				for (j = 0; j < k; j++)
				{
					y = g[j] * wn[j] % p;
					g[j] = f[j] + p - y;
					f[j] += y;
				}
			}
			if (__lg(n / k) % T == 1) for (lll &x : a) x %= p;
		}
		if (o)
		{
			y = ksm(n, p - 2);
			for (lll &x : a) x = x * y % p;
			reverse(1 + all(a));
		}
	}
	ull &operator[](const int &x) { return a[x]; }
	const ull &operator[](const int &x) const { return a[x]; }
	static void plus_by(vector<ull> &a, const vector<ull> &b)
	{
		int n = a.size(), m = b.size(), i, j;
		cmax(n, m);
		a.resize(++n);
		for (i = 0; i < m; i++) if ((a[i] += b[i]) >= base) a[i] -= base, ++a[i + 1];
		for (i = m; i < n && a[i] >= base; i++) a[i] -= base, ++a[i + 1];
		if (a[n - 1] == 0) a.pop_back();
	}
	static void minus_by(vector<ull> &a, const vector<ull> &b)
	{
		int n = a.size(), m = b.size(), i, j;
		for (i = 0; i < m; i++) if (!(a[i] & sign) && a[i] >= b[i]) a[i] -= b[i];
		else --a[i + 1], a[i] += base - b[i];
		for (; i < n && (a[i] & sign); i++) a[i] += base, --a[i + 1];
		while (a.size() > 1 && !a.back()) a.pop_back();
	}
	static bool less(const vector<ull> &a, const vector<ull> &b)
	{
		if (a.size() != b.size()) return a.size() < b.size();
		for (int i = a.size() - 1; i >= 0; i--) if (a[i] != b[i]) return a[i] < b[i];
		return 0;
	}
	static int cal(int x) { return 1 << __lg(max(x, 1) * 2 - 1); }
public:
	bigint &operator+=(const bigint &o)
	{
		if (neg == o.neg) plus_by(a, o.a);
		else if (neg)
		{
			if (less(o.a, a)) minus_by(a, o.a);
			else
			{
				neg = 0;
				auto t = o.a;
				swap(a, t);
				minus_by(a, t);
			}
		}
		else
		{
			if (less(a, o.a))
			{
				neg = 1;
				auto t = o.a;
				swap(a, t);
				minus_by(a, t);
			}
			else minus_by(a, o.a);
		}
		return *this;
	}
	bigint &operator-=(const bigint &o)
	{
		neg ^= 1;
		*this += o;
		neg ^= 1;
		if (a == vector<ull>{0}) neg = 0;
		return *this;
	}
	bigint &operator*=(const bigint &o)
	{
		neg ^= o.neg;
		int n = a.size(), m = o.a.size(), i, j;
		assert(min(n, m) <= p / ((base - 1) * (base - 1)));
		if (min(n, m) <= 64 && 0)
		{
			vector<ull> c(n + m);
			for (i = 0; i < n; i++) for (j = 0; j < m; j++) c[i + j] += a[i] * o[j];
			for (i = 0; i < n + m - 1; i++)
			{
				c[i + 1] += c[i] / base;
				c[i] %= base;
			}
			swap(a, c);
			while (a.size() > 1 && !a.back()) a.pop_back();
			if (a == vector<ull>{0}) neg = 0;
			return *this;
		}
		int len = cal(n + m);
		vector<lll> f(len), g(len);
		copy_n(a.begin(), n, f.begin());
		copy_n(o.a.begin(), m, g.begin());
		dft(f); dft(g);
		for (i = 0; i < len; i++) f[i] = f[i] * g[i] % p;
		dft(f, 1);
		a.resize(n + m);
		copy_n(f.begin(), n + m - 1, a.begin());
		for (i = n + m - 2; i >= 0; i--)
		{
			a[i + 1] += a[i] / base;
			a[i] %= base;
		}
		for (i = 0; i < n + m - 1; i++)
		{
			a[i + 1] += a[i] / base;
			a[i] %= base;
		}
		while (a.size() > 1 && !a.back()) a.pop_back();
		if (a == vector<ull>{0}) neg = 0;
		return *this;
	}
	bigint &operator/=(long long x)//to zero
	{
		if (x < 0) x = -x, neg ^= 1;
		for (int i = a.size() - 1; i; i--)
		{
			a[i - 1] += a[i] % x * base;
			a[i] /= x;
		}
		a[0] /= x;
		while (a.size() > 1 && !a.back()) a.pop_back();
		if (a == vector<ull>{0}) neg = 0;
		return *this;
	}
	bigint operator+(bigint o) const { return o += *this; }
	bigint operator-(bigint o) const { o -= *this; if (o.a != vector<ull>{0}) o.neg ^= 1; return o; }
	bigint operator*(bigint o) const { return o *= *this; }
	bigint operator/(long long x) const { auto res = *this; return res /= x; }
	long long operator%(long long x) const
	{
		bool flg = neg;
		if (x < 0) flg ^= 1, x = -x;
		ull res = 0;
		for (int i = (base % x == 0 ? 0 : a.size() - 1); i >= 0; i--) res = (res * base + a[i]) % x;
		return (long long)res * (flg ? -1 : 1);
	}
	bigint(long long x = 0) :neg(0)
	{
		if (x < 0) x = -x, neg = 1;
		a.push_back(x % base);
		while (x /= base) a.push_back(x % base);
	}
	bool operator<(const bigint &o) const { return cmp(*this, o) < 0; }
	bool operator>(const bigint &o) const { return cmp(*this, o) > 0; }
	bool operator<=(const bigint &o) const { return cmp(*this, o) <= 0; }
	bool operator>=(const bigint &o) const { return cmp(*this, o) >= 0; }
	bool operator==(const bigint &o) const { return cmp(*this, o) == 0; }
	bool operator!=(const bigint &o) const { return cmp(*this, o) != 0; }
};
int cmp(const bigint &a, const bigint &b)
{
	if (a.neg != b.neg) return a.neg ? -1 : 1;
	if (a.neg) return -cmp(b, a);
	if (a.a.size() != b.a.size()) return a.a.size() < b.a.size() ? -1 : 1;
	for (int i = a.a.size() - 1; i >= 0; i--) if (a.a[i] != b.a[i]) return a.a[i] < b.a[i] ? -1 : 1;
	return 0;
}
istream &operator>>(istream &cin, bigint &x)
{
	x.neg = 0;
	x.a.clear();
	string s;
	cin >> s;
	const static int length = bigint::length;
	static int mp[128], _ = [&]() {
		for (int i = '0'; i <= '9'; i++) mp[i] = i - '0';
		for (int i = 'a'; i <= 'z'; i++) mp[i] = i - 'a' + 10;
		for (int i = 'A'; i <= 'Z'; i++) mp[i] = i - 'A' + 10;
		return 0;
	}();
	reverse(all(s));
	if (s.back() == '-') x.neg = 1, s.pop_back();
	ull base = 1;
	for (int i = 0; i < s.size(); i++)
	{
		if (i % length == 0) x.a.push_back(0), base = 1;
		x.a.back() += mp[s[i]] * base;
		base *= bigint::output_base;
	}
	return cin;
}
ostream &operator<<(ostream &cout, const bigint &x)
{
	if (x.neg) cout << "-";
	const static int length = bigint::length;
	if (bigint::output_base == 10)
	{
		cout << setfill('0') << x.a.back();
		for (int i = (int)x.a.size() - 2; i >= 0; i--) cout << setw(length) << x.a[i];
	}
	else if (bigint::output_base == 16)
	{
		cout << hex << uppercase << setfill('0') << x.a.back();
		for (int i = (int)x.a.size() - 2; i >= 0; i--) cout << setw(length) << x.a[i];
		cout << dec;
	}
	else assert(0);
	return cout;
}
bigint abs(bigint x)
{
	x.neg = 0;
	return x;
}
bigint gcd(bigint x, bigint y)
{
	x.neg = y.neg = 0;
	if (x == bigint(0)) return y;
	if (y == bigint(0)) return x;
	int c1 = 0, c2 = 0;
	while (x % 2 == 0) x /= 2, ++c1;
	while (y % 2 == 0) y /= 2, ++c2;
	cmin(c1, c2);
	if (x > y) swap(x, y);
	while (x != y)
	{
		y -= x;
		y /= 2;
		while (y % 2 == 0) y /= 2;
		if (x > y) swap(x, y);
	}
	while (c1--) y *= bigint(2);
	return y;
}
bigint::lll bigint::w[bigint::N];
int bigint::r[bigint::N];
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(0);
	int T; cin >> T;
	while (T--)
	{
		bigint a, b;
		cin >> a >> b;
		cout << (a *= b) << '\n';
	}
}
