
#include <optional>
namespace NTT
{
	using ull = unsigned long long;
	const ull g = 3, p = 998244353;
	const int N = 1 << 21;//务必修改
	ull inv[N + 1], fac[N + 1], ifac[N + 1];//非必要
	int _ = []() {//非必要
		ifac[0] = fac[0] = fac[1] = ifac[1] = inv[1] = 1;
		for (int i = 2, j; i <= N; i++)
		{
			j = p / i;
			inv[i] = (p - j) * inv[p - i * j] % p;
			fac[i] = fac[i - 1] * i % p;
			ifac[i] = ifac[i - 1] * inv[i] % p;
		}
		return 0;
	}();
	ull w[N];
	int r[N];
	ull ksm(ull x, ull y)
	{
		if (y == p - 2 && x < N) return inv[x];
		ull r = 1;
		while (y)
		{
			if (y & 1) r = r * x % p;
			x = x * x % p;
			y >>= 1;
		}
		return r;
	}
	void init(int n)
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
	int cal(int x) { return 1 << __lg(max(x, 1) * 2 - 1); }
	struct Q :vector<ull>
	{
		bool flag;
		Q &operator%=(int n) { assert((n & -n) == n); resize(n); return *this; }
		Q operator%(int n) const
		{
			assert((n & -n) == n);
			if (size() <= n)
			{
				auto f = *this;
				return f %= n;
			}
			return Q(vector(begin(), begin() + n));
		}
		int deg() const
		{
			int n = size() - 1;
			while (n >= 0 && begin()[n] == 0) --n;
			return n;
		}
		explicit Q(int x = 1, bool f = 0) :flag(f), vector<ull>(cal(x)) { }//小心：{}会调用这条而非下一条
		Q(const vector<ull> &o, bool f = 0) :Q(o.size(), f) { copy(all(o), begin()); }
		Q(const initializer_list<ull> &o, bool f = 0) :Q(vector(o), f) { }
		ull fx(ull x)
		{
			ull r = 0;
			for (auto it = rbegin(); it != rend(); ++it) r = (r * x + *it) % p;
			return r;
		}
		void dft()
		{
			int n = size(), i, j, k;
			ull y, *f, *g, *wn, *a = data();
			init(n);
			for (i = 1; i < n; i++) if (i < r[i]) ::swap(a[i], a[r[i]]);
			for (k = 1; k < n; k *= 2)
			{
				wn = w + k;
				for (i = 0; i < n; i += k * 2)
				{
					g = (f = a + i) + k;
					for (j = 0; j < k; j++)
					{
						y = g[j] * wn[j] % p;
						g[j] = f[j] + p - y;
						f[j] += y;
					}
				}//此处要求 14*p*p<=2^64。如果调整模数，需要修改 12。
				if (__lg(n / k) % 12 == 1) for (i = 0; i < n; i++) a[i] %= p;
			}
			if (flag)
			{
				y = ksm(n, p - 2);
				for (i = 0; i < n; i++) a[i] = a[i] * y % p;
				reverse(a + 1, a + n);
			}
			flag ^= 1;
		}
		void hf_dft()
		{
			assert(size() >= 2 && flag);
			int n = size() / 2, i, j, k;
			ull x, y, *f, *g, *wn, *a = data();
			init(n);
			for (i = 1; i < n; i++) if (i < r[i]) ::swap(a[i], a[r[i]]);
			for (k = 1; k < n; k *= 2)
			{
				wn = w + k;
				for (i = 0; i < n; i += k * 2)
				{
					g = (f = a + i) + k;
					for (j = 0; j < k; j++)
					{
						y = g[j] * wn[j] % p;
						g[j] = f[j] + p - y;
						f[j] += y;
					}
				}
				if (__lg(n / k) % 12 == 1) for (i = 0; i < n; i++) a[i] %= p;
			}
			if (flag)
			{
				x = ksm(n, p - 2);
				for (i = 0; i < n; i++) a[i] = a[i] * x % p;
				reverse(a + 1, a + n);
			}
			flag ^= 1;
		}
		Q operator<<(int m) const
		{
			int n = deg(), i;
			Q r(n + m + 1);
			for (i = 0; i <= n; i++) r[i + m] = at(i);
			return r;
		}
		Q operator>>(int m) const
		{
			int n = deg(), i;
			if (n < m) return Q();
			Q r(n + 1 - m);
			for (i = m; i <= n; i++) r[i - m] = at(i);
			return r;
		}
	};
	Q shrink(Q f) { return f %= cal(f.deg() + 1); }
	ostream &operator<<(ostream &cout, const Q &o)
	{
		int n = o.deg();
		if (n < 0) return cout << "[0]";
		cout << "[" << o[n];
		for (int i = n - 1; i >= 0; i--) cout << ", " << o[i];
		return cout << "]";
	}
	Q der(const Q &f)
	{
		ull n = f.size(), i;
		Q r(n);
		for (i = 1; i < n; i++) r[i - 1] = f[i] * i % p;
		return r;
	}
	Q integral(const Q &f)
	{
		ull n = f.size(), i;
		Q r(n);
		for (i = 1; i < n; i++) r[i] = f[i - 1] * inv[i] % p;
		return r;
	}
	Q &operator+=(Q &f, ull x) { (f[0] += x) %= p; return f; }
	Q operator+(Q f, ull x) { return f += x; }
	Q &operator-=(Q &f, ull x) { (f[0] += p - x) %= p; return f; }
	Q operator-(Q f, ull x) { return f -= x; }
	Q &operator*=(Q &f, ull x) { for (ull &y : f) (y *= x) %= p; return f; }
	Q operator*(Q f, ull x) { return f *= x; }
	Q &operator+=(Q &f, const Q &g)
	{
		f %= max(f.size(), g.size());
		for (int i = 0; i < g.size(); i++) f[i] = (f[i] + g[i]) % p;
		return f;
	}
	Q operator+(Q f, const Q &g) { return f += g; }
	Q &operator-=(Q &f, const Q &g)
	{
		f %= max(f.size(), g.size());
		for (int i = 0; i < g.size(); i++) f[i] = (f[i] + p - g[i]) % p;
		return f;
	}
	Q operator-(Q f, const Q &g) { return f -= g; }
	Q &operator*=(Q &f, Q g)//卷积
	{
		if (f.flag | g.flag)
		{
			int n = f.size(), i;
			assert(n == g.size());
			if (!f.flag) f.dft();
			if (!g.flag) g.dft();
			for (i = 0; i < n; i++) (f[i] *= g[i]) %= p;
			f.dft();
		}
		else
		{
			int n = cal(f.size() + g.size() - 1), i, j;
			int m1 = f.deg(), m2 = g.deg();
			if ((ull)m1 * m2 > (ull)n * __lg(n) * 8)
			{
				(f %= n).dft(); (g %= n).dft();
				for (i = 0; i < n; i++) (f[i] *= g[i]) %= p;
				f.dft();
			}
			else
			{
				vector<ull> r(max(0, m1 + m2 + 1));
				for (i = 0; i <= m1; i++) for (j = 0; j <= m2; j++) (r[i + j] += f[i] * g[j]) %= p;
				f = Q(n);
				copy(all(r), f.begin());
			}
		}
		return f;
	}
	Q operator*(Q f, const Q &g) { return f *= g; }
	Q &operator&=(Q &f, Q g)//循环卷积
	{
		assert(f.size() == g.size());
		int n = f.size(), i;
		if (!f.flag) f.dft();
		if (!g.flag) g.dft();
		for (i = 0; i < n; i++) (f[i] *= g[i]) %= p;
		f.dft();
		return f;
	}
	Q operator&(Q f, const Q &g) { return f &= g; }
	Q &operator^=(Q &f, Q g)//差卷积
	{
		int n = f.size();
		g %= n;
		reverse(all(g));
		f *= g;
		rotate(f.begin(), n - 1 + all(f));
		return f %= n;
	}
	Q operator^(Q f, const Q &g) { return f ^= g; }
	Q sqr(Q f)
	{
		assert(!f.flag);
		int n = f.size() * 2, i;
		(f %= n).dft();
		for (i = 0; i < n; i++) f[i] = f[i] * f[i] % p;
		f.dft();
		return f;
	}
	/*Q operator~(const Q &f)
	{
		Q r;
		r[0]=ksm(f[0],p-2);
		for (int i=1; i<=f.size(); i*=2) r=(-((f%i)*r-2)*r)%i;
		return r;
	}//trivial, 5e5 750ms*/
	Q operator~(const Q &f)
	{
		Q q, r, g;
		int n = f.size(), i, j, k;
		r[0] = ksm(f[0], p - 2);
		for (j = 2; j <= n; j *= 2)
		{
			k = j / 2;
			g = (r %= j) % k;
			r.dft();
			q = f % j * r;
			fill_n(q.begin(), k, 0);
			r *= q;
			copy(all(g), r.begin());
			for (i = k; i < j; i++) r[i] = (p - r[i]) % p;
		}
		return r;
	}//5e5 200ms, inv(1 6 3 4 9)=(1 998244347 33 998244169 1020)
	Q &operator/=(Q &f, const Q &g) { int n = f.size(); return (f *= ~g) %= n; }
	Q operator/(Q f, const Q &g) { return f /= g; }
	void cdq(Q &f, Q &g, int l, int r)//g_0=1,i*g_i=g_{i-j}*f_j,use for cdq
	{
		static vector<Q> cd;
		int i, m = l + r >> 1, n = r - l, nn = n >> 1;
		if (r - l == f.size())
		{
			g = Q(n);
			cd.clear();
			for (i = 2; i <= n; i *= 2)
			{
				cd.emplace_back(i);
				Q &h = cd.back();
				h %= i;
				copy_n(f.begin(), i, h.begin());
				h.dft();
			}
		}
		if (l + 1 == r)
		{
			g[l] = l ? g[l] * inv[l] % p : 1;
			return;
		}
		cdq(f, g, l, m);
		Q h(n);
		copy_n(g.begin() + l, nn, h.begin());
		h *= cd[__lg(n) - 1];
		for (i = m; i < r; i++) (g[i] += h[i - l]) %= p;
		cdq(f, g, m, r);
	}
	Q exp_cdq(Q f)
	{
		Q g;
		int n = f.size(), i;
		for (i = 1; i < n; i++) f[i] = f[i] * i % p;
		cdq(f, g, 0, n);
		return g;
	}//5e5 455ms
	Q ln(const Q &f) { return integral(der(f) / f); }
	//5e5 330ms, ln(1 2 3 4 5)=(0 2 1 665496236 499122177)
	Q exp(Q f)
	{
		Q r; r[0] = 1;
		for (int i = 1; i <= f.size(); i *= 2) (r *= f % i - ln(r % i) + 1) %= i;
		return r;
	}//5e5 700ms, exp(0 4 2 3 5)=(1 4 10 665496257 665496281)
	Q exp_new(Q b)
	{
		Q h, f, r, u, v, bj;
		int n = b.size(), i, j, k;
		r[0] = h[0] = 1;
		for (j = 2; j <= n; j *= 2)
		{
			f = bj = der(b % j); k = j / 2; fill(k + all(bj), 0);
			h.dft(); u = der(r) & h;
			v = (r & h) % j - 1 & bj;
			for (i = 0; i < k; i++) f[i + k] = (p * p + u[i] - v[i] - f[i] - f[i + k]) % p, f[i] = 0;
			f[k - 1] = (f[j - 1] + v[k - 1]) % p;
			u = (r %= j) & integral(f);
			for (i = k; i < j; i++) r[i] = (p - u[i]) % p;
			if (j < n) h = ~r;
		}
		return r;
	}//5e5 420ms
	optional<ull> mosqrt(ull x)
	{
		static mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
		static ull W;
		struct P
		{
			ull x, y;
			P operator*(const P &a) const
			{
				return {(x * a.x + y * a.y % p * W) % p, (x * a.y + y * a.x) % p};
			}
		};
		if (x == 0) return {0};
		if (ksm(x, p - 1 >> 1) != 1) return { };
		ull y;
		do y = rnd() % p; while (ksm(W = (y * y % p + p - x) % p, p - 1 >> 1) <= 1);//not for p=2
		y = [&](P x, ull y) {
			P r{1, 0};
			while (y)
			{
				if (y & 1) r = r * x;
				x = x * x; y >>= 1;
			}
			return r.x;
		}({y, 1}, p + 1 >> 1);
		return {y * 2 < p ? y : p - y};
	}
	optional<Q> sqrt(Q f)
	{
		const static ull i2 = p + 1 >> 1;
		Q r;
		int n = f.size(), i, l;

		for (i = 0; i < n; i++) if (f[i]) break;
		if (i == n) return f;
		if (i & 1) return { };
		l = i / 2;
		copy(i + all(f), f.begin());
		fill(n - i + all(f), 0);

		auto rt = mosqrt(f[0]);
		if (rt) r[0] = rt.value(); else return { };
		for (i = 2; i <= n; i *= 2) r = (sqr(r) + f % i) / (r % i) % i * i2;

		copy_backward(all(r) - l, r.end());
		fill_n(r.begin(), l, 0);

		return {r};
	}//5e5 530ms, sqrt(0 0 4 2 3)=(0 2 499122177 311951361 171573248)
	optional<Q> sqrt_new(Q f)
	{
		const static ull i2 = p + 1 >> 1;
		Q q, r;
		int n = f.size(), i, j, k, l;

		for (i = 0; i < n; i++) if (f[i]) break;
		if (i == n) return f;
		if (i & 1) return { };
		l = i / 2;
		copy(i + all(f), f.begin());
		fill(n - i + all(f), 0);

		auto rt = mosqrt(f[0]);
		if (rt) r[0] = rt.value(); else return { };
		for (j = 2; j <= n; j *= 2)
		{
			k = j / 2; (q = r).dft(); (q &= q) %= j;
			for (i = k; i < j; i++) q[i] = (q[i - k] + p * 2 - f[i] - f[i - k]) * i2 % p, q[i - k] = 0;
			q &= ~r % j; r %= j;
			for (i = k; i < j; i++) r[i] = (p - q[i]) % p;
		}

		copy_backward(all(r) - l, r.end());
		fill_n(r.begin(), l, 0);

		return {r};
	}//5e5 280ms
	Q pow(Q b, ull m)//不应传入超过 int 内容
	{
		assert(m <= 1llu << 32);
		int n = b.size(), i, j = n, k;
		for (i = 0; i < n; i++) if (b[i]) { j = i; break; }
		if (j == n) return b[0] = !m, b;
		if (j * m >= n) return Q(n);
		copy(j + all(b), b.begin());
		fill(n - j + all(b), 0);
		k = b[0]; j *= m;
		b = exp_new(ln(b * ksm(k, p - 2)) * m) * ksm(k, m);
		copy_backward(all(b) - j, b.end());
		fill_n(b.begin(), j, 0);
		return b;
	}
	Q pow(Q b, string s)
	{
		int n = b.size(), i, j = n, k;
		for (i = 0; i < n; i++) if (b[i]) { j = i; break; }
		if (j == n) return b[0] = s == "0", b;
		if (j && (s.size() > 8 || j * stoll(s) >= n)) return Q(n);
		ull m0 = 0, m1 = 0;
		for (auto c : s) m0 = (m0 * 10 + c - '0') % p, m1 = (m1 * 10 + c - '0') % (p - 1);
		copy(j + all(b), b.begin());
		fill(n - j + all(b), 0);
		k = b[0]; j *= m0;
		b = exp_new(ln(b * ksm(k, p - 2)) * m0) * ksm(k, m1);
		copy_backward(all(b) - j, b.end());
		fill_n(b.begin(), j, 0);
		return b;
	}//5e5 1e18 700ms
	Q pow2(Q b, ull m)
	{
		int n = b.size();
		Q r(n); r[0] = 1;
		while (m)
		{
			if (m & 1) (r *= b) %= n;
			if (m >>= 1) b = sqr(b) % n;
		}
		return r;
	}//5e5 1e18 7425ms
	Q div(Q f, Q g)
	{
		int n = 0, m = 0, i;
		for (i = f.size() - 1; i >= 0; i--) if (f[i]) { n = i + 1; break; }
		for (i = g.size() - 1; i >= 0; i--) if (g[i]) { m = i + 1; break; }
		assert(m);
		if (n < m) return Q(1);
		reverse(f.begin(), f.begin() + n);
		reverse(g.begin(), g.begin() + m);
		n = n - m + 1; m = cal(n);
		f = (f % m) / (g % m) % m;
		fill(n + all(f), 0);
		reverse(f.begin(), f.begin() + n);
		return f;
	}
	Q mod(const Q &a, const Q &b)
	{
		if (a.deg() < b.deg()) return shrink(a);
		Q r = (a - b * div(a, b));
		return shrink(r %= min(r.size(), b.size()));
	}
	Q pow(Q x, ull y, Q f)
	{
		Q r(1);
		r[0] = 1;
		while (y)
		{
			if (y & 1) r = mod(r * x, f);
			if (y >>= 1) x = mod(sqr(x), f);
		}
		return r;
	}
	pair<Q, Q> div_mod(Q a, Q b)
	{
		if (b.size() <= 16)
		{
			int n = a.deg(), m = b.deg(), i, j;
			if (m == 0)
			{
				assert(b[0]);
				a *= ksm(b[0], p - 2);
				return {a, Q()};
			}
			assert(m > 0);
			Q q(n - m + 1);
			ull k = b[m], ik = ksm(k, p - 2);
			for (i = 0; i < m; i++) (b[i] *= ik) %= p;
			for (i = n - m; i >= 0; i--)
			{
				q[i] = a[i + m];
				for (j = 0; j <= m; j++) (a[i + j] += (p - q[i]) * b[j]) %= p;
				(q[i] *= ik) %= p;
			}
			return {q, a % min<ull>(a.size(), cal(m))};
		}
		Q q = div(a, b);
		(a -= (b *= q));
		return {q, a %= min(a.size(), b.size())};
	}
	//5e5 430ms (1 2 3 4)=(916755018 427819009)*(5 6 7)+(407446676 346329673)
	// Q cdq_inv(const Q &f) { return (~(f-1))*(p-1); }//g_0=1,g_i=g_{i-j}*f_j ?
	ull recurrent(const vector<ull> &f, const vector<ull> &a, ull m)//常系数齐次线性递推，find a_m,a_n=a_{n-i}*f_i,f_1...k,a_0...k-1
	{
		if (m < a.size()) return a[m];
		assert(f.size() == a.size() + 1 && f[0] == 0);
		int k = a.size(), n = cal(k + 1) * 2, i;
		ull ans = 0;
		Q h(n), g(2);
		for (i = 1; i <= k; i++) h[k - i] = (p - f[i]) % p;
		h[k] = g[1] = 1;
		Q r = pow(g, m, h);
		k = min(k, (int)r.size());
		for (i = 0; i < k; i++) ans = (ans + a[i] * r[i]) % p;
		return ans;
	}//1e5 1e18 8500ms
	ull recurrent_new(const vector<ull> &f, const vector<ull> &a, ull m)//常系数齐次线性递推，find a_m,a_n=a_{n-i}*f_i,f_1...k,a_0...k-1
	{
		const static ull i2 = p + 1 >> 1;
		if (m < a.size()) return a[m];
		assert(f.size() == a.size() + 1 && f[0] == 0);
		int k = a.size(), n = cal(k + 1), i;
		Q g(n * 2), h(n * 2);
		for (h[0] = i = 1; i <= k; i++) h[i] = (p - f[i]) % p;
		copy(all(a), g.begin());
		g &= h; fill(k++ + all(g), 0);
		vector<ull> res(n);
		while (m)
		{
			if (m & 1)
			{
				ull x = p - g[0];
				for (i = 1; i < k; i += 2) res[i >> 1] = x * h[i] % p;
				copy_n(g.begin() + 1, k - 1, g.begin());
				g[k - 1] = 0;
			}
			g.dft(); h.dft();
			ull *a = g.data(), *b = h.data(), *c = a + n, *d = b + n;
			for (i = 0; i < n; i++) g[i] = (a[i] * d[i] + b[i] * c[i]) % p * i2 % p;
			for (i = 0; i < n; i++) h[i] = h[i] * h[i ^ n] % p;
			g.hf_dft(); h.hf_dft();
			fill(k + all(g), 0);
			if (m & 1) for (i = 0; i < k; i++) (g[i] += res[i]) %= p;
			fill(k + all(h), 0);
			m >>= 1;
		}
		assert(h[0] == 1);
		return g[0];
	}//1e5 1e18 1000ms
	vector<ull> recurrent_interval(const vector<ull> &f, const vector<ull> &a, ull L, ull R)//常系数齐次线性递推，find a_[L,R),a_n=a_{n-i}*f_i,f_1...k,a_0...k-1
	{
		assert(f.size() == a.size() + 1 && f[0] == 0);
		int k = a.size(), n = cal(k + 1) * 2, i, len = R - L;
		ull ans = 0, m = L;
		Q h(n), g(2), r;
		for (i = 1; i <= k; i++) h[k - i] = (p - f[i]) % p;
		h[k] = g[1] = r[0] = 1;
		while (m)
		{
			if (m & 1) r = mod(r * g, h);
			if (m >>= 1) g = mod(sqr(g), h);
		}
		Q F(f), A(a);
		F[0] = p - 1;
		A *= F;
		A %= cal(k);
		fill(k + all(A), 0);
		n = cal(len + k);
		F %= n;
		A *= ~F;
		r %= cal(k);
		reverse(r.begin(), r.begin() + k);
		r *= A;
		r.erase(r.begin(), r.begin() + k - 1);
		r.resize(len);
		return r;
	}//1e5 1e18 5e5 10000ms
	Q prod(const vector<Q> &a)
	{
		if (!a.size()) return {1};
		function<Q(int, int)> dfs = [&](int l, int r) {
			if (r - l == 1) return a[l];
			int m = l + r >> 1;
			return shrink(dfs(l, m) * dfs(m, r));
		};
		return dfs(0, a.size());
	}
	Q prod_new(const vector<Q> &a)
	{
		if (!a.size()) return {1};
		struct cmp
		{
			bool operator()(const Q &f, const Q &g) const { return f.size() > g.size(); }
		};
		priority_queue<Q, vector<Q>, cmp> q(all(a));
		while (q.size() > 1)
		{
			auto f = q.top(); q.pop();
			f = shrink(f * q.top()); q.pop();
			q.push(f);
		}
		return q.top();
	}
	vector<ull> evaluation(const Q &f, const vector<ull> &X)
	{
		int m = X.size(), n = f.size() - 1, i, j;
		vector<Q> pro(m * 4 + 4);
		while (n > 1 && !f[n]) --n;
		vector<ull> y(m);
		function<void(int, int, int)> build = [&](int x, int l, int r) {
			if (l + 1 == r)
			{
				pro[x] = Q(vector{(p - X[l]) % p, 1llu});
				return;
			}
			int mid = l + r >> 1, c = x * 2;
			build(c, l, mid); build(c + 1, mid, r);
			pro[x] = shrink(pro[c] * pro[c + 1]);
		};
		function<void(int, int, int, Q, int)> dfs = [&](int x, int l, int r, Q f, int d) {
			const static int limit = 256;
			if (d >= r - l) f = shrink(mod(f, pro[x]));
			if (r - l < limit)
			{
				for (int i = l; i < r; i++) y[i] = f.fx(X[i]);
				return;
			}
			int mid = l + r >> 1, c = x * 2;
			dfs(c, l, mid, f, d);
			dfs(c + 1, mid, r, f, d);
		};
		build(1, 0, m);
		dfs(1, 0, m, f, n);
		return y;
	}//131072 880ms
	vector<ull> evaluation_new(Q f, const vector<ull> &X)//多项式多点求值
	{
		int m = X.size(), i, j;
		vector<ull> y(m);
		if (X.size() <= 10)
		{
			for (i = 0; i < m; i++) y[i] = f.fx(X[i]);
			return y;
		}
		int n = f.size();
		while (n > 1 && !f[n - 1]) --n;
		f.resize(cal(n));
		vector<Q> pro(m * 4 + 4);
		function<void(int, int, int)> build = [&](int x, int l, int r) {
			if (l == r)
			{
				pro[x] = Q(vector{1llu, (p - X[l]) % p});
				return;
			}
			int m = l + r >> 1, c = x * 2;
			build(c, l, m); build(c + 1, m + 1, r);
			pro[x] = shrink(pro[c] * pro[c + 1]);
		};
		function<void(int, int, int, Q)> dfs = [&](int x, int l, int r, Q f) {
			const static int limit = 30;
			if (r - l + 1 <= limit)
			{
				int m = r - l + 1, m1, m2, mid = l + r >> 1, i, j, k;
				static ull g[limit + 2], g1[limit + 2], g2[limit + 2];
				m1 = m2 = r - l;
				copy_n(f.data(), m, g1);
				copy_n(g1, m, g2);
				for (i = mid + 1; i <= r; i++, --m1) for (k = 0; k < m1; k++) g1[k] = (g1[k] + g1[k + 1] * (p - X[i])) % p;
				for (i = l; i <= mid; i++, --m2) for (k = 0; k < m2; k++) g2[k] = (g2[k] + g2[k + 1] * (p - X[i])) % p;
				for (i = l; i <= mid; i++)
				{
					copy_n(g1, (m = m1) + 1, g);
					for (j = l; j <= mid; j++) if (i != j)
					{
						for (k = 0; k < m; k++) g[k] = (g[k] + g[k + 1] * (p - X[j])) % p;
						--m;
					}
					y[i] = g[0];
				}
				for (i = mid + 1; i <= r; i++)
				{
					copy_n(g2, (m = m2) + 1, g);
					for (j = mid + 1; j <= r; j++) if (i != j)
					{
						for (k = 0; k < m; k++) g[k] = (g[k] + g[k + 1] * (p - X[j])) % p;
						--m;
					}
					y[i] = g[0];
				}
				return;
			}
			int mid = l + r >> 1, c = x * 2, n = f.size();
			f.dft();
			for (auto [x, len] : {pair{c, r - mid}, {c + 1, mid - l + 1}})
			{
				pro[x] %= n;
				reverse(all(pro[x])); pro[x] &= f;
				rotate(all(pro[x]) - 1, pro[x].end());
				pro[x] %= cal(len);
				fill(len + all(pro[x]), 0);
			}
			dfs(c, l, mid, pro[c + 1]);
			dfs(c + 1, mid + 1, r, pro[c]);
		};
		build(1, 0, m - 1);
		pro[1] %= f.size();
		(f ^= ~pro[1]) %= cal(m);
		fill(min(m, n) + all(f), 0);
		dfs(1, 0, m - 1, f);
		return y;
	}//131072 460ms
	ull factorial(ull n)
	{
		if (n >= p) return 0;
		if (n <= 1) return 1 % p;
		ull B = ::sqrt(n), i;
		vector F(B, Q({0, 1}));
		for (i = 0; i < B; i++) F[i][0] = i + 1;
		auto f = prod(F);
		vector<ull> x(B);
		for (i = 0; i < B; i++) x[i] = i * B;
		ull r = 1;
		auto y = evaluation(f, x);
		for (i = 0; i < B; i++) r = r * y[i] % p;
		for (i = B * B + 1; i <= n; i++) r = r * i % p;
		return r;
	}//998244352 170ms
	vector<ull> getinvs(vector<ull> a)
	{
		int n = a.size(), i;
		if (n <= 2)
		{
			for (i = 0; i < n; i++) a[i] = ksm(a[i], p - 2);
			return a;
		}
		vector<ull> l(n), r(n);
		l[0] = a[0]; r[n - 1] = a[n - 1];
		for (i = 1; i < n; i++) l[i] = l[i - 1] * a[i] % p;
		for (i = n - 2; i; i--) r[i] = r[i + 1] * a[i] % p;
		ull x = ksm(l[n - 1], p - 2);
		a[0] = x * r[1] % p; a[n - 1] = x * l[n - 2] % p;
		for (i = 1; i < n - 1; i++) a[i] = x * l[i - 1] % p * r[i + 1] % p;
		return a;
	}
	Q interpolation(const vector<ull> &X, const vector<ull> &y)//多项式快速插值
	{
		assert(X.size() == y.size());
		int n = X.size(), i, j;
		if (n <= 1) return Q(y);
		if (1)
		{
			auto vv = X; sort(all(vv));
			assert(unique(all(vv)) - vv.begin() == n);
		}
		vector<Q> sum(4 * n + 4), pro(4 * n + 4);
		function<void(int, int, int)> build = [&](int x, int l, int r) {
			if (l == r)
			{
				sum[x] = Q(vector{(p - X[l]) % p, 1llu});
				return;
			}
			int mid = l + r >> 1, c = x * 2;
			build(c, l, mid); build(c + 1, mid + 1, r);
			sum[x] = shrink(sum[c] * sum[c + 1]);
		};
		build(1, 0, n - 1);
		auto v = evaluation_new(sum[1] = der(sum[1]), X);
		assert(v.size() == n);
		auto Y = getinvs(v);
		for (i = 0; i < n; i++) Y[i] = Y[i] * y[i] % p;
		function<void(int, int, int)> dfs = [&](int x, int l, int r) {
			if (l == r)
			{
				pro[x][0] = Y[l];
				return;
			}
			int c = x * 2, mid = l + r >> 1;
			dfs(c, l, mid); dfs(c | 1, mid + 1, r);
			pro[x] = shrink((pro[c] * sum[c | 1]) + (pro[c | 1] * sum[c]));
		};
		dfs(1, 0, n - 1);
		return pro[1] %= cal(n);
	}//131072 1150ms
	Q comp(const Q &f, Q g)//多项式复合 f(g(x))=[x^i]f(x)g(x)^i
	{
		int n = f.size(), l = ceil(::sqrt(n)), i, j;
		assert(n >= g.size());//返回 n-1 次多项式
		vector<Q> a(l + 1), b(l);
		a[0] %= n; a[0][0] = 1; a[1] = g;
		g %= n * 2;
		Q u = g, v(n);
		g.dft();
		for (i = 2; i <= l; i++) a[i] = ((u &= g) %= n), u %= n * 2;
		for (i = 2; i < l; i++)
		{
			u.dft(); b[i - 1] = u;
			u &= b[1]; fill(n + all(u), 0);
		}
		u.dft(); b[l - 1] = u;
		for (i = 0; i < l; i++)
		{
			fill(all(v), 0);
			for (j = 0; j < l; j++) if (i * l + j < n) v += a[j] * f[i * l + j];
			if (i == 0) u = v; else u += ((v %= n * 2) &= b[i]) %= n;
		}
		return u;
	}//n^2+n\sqrt n\log n，8000 350ms
	Q comp_inv(Q f)//多项式复合逆 g(f(x))=x，求 g，[x^n]g=([x^{n-1}](x/f)^n)/n，要求常数 0 一次非 0
	{
		assert(!f[0] && f[1]);
		int n = f.size(), l = ceil(::sqrt(n)), i, j, k, m;//l>=2
		rotate(f.begin(), 1 + all(f));
		f = ~f;
		vector<Q> a(l + 1), b(l);
		Q u, v;
		u = a[1] = f;
		u %= n * 2; (v = u).dft();
		for (i = 2; i <= l; i++)
		{
			u &= v;
			fill(n + all(u), 0);
			a[i] = u;
		}
		b[0] %= n; b[0][0] = 1; b[1] = u; (v = u).dft();
		for (i = 2; i < l; i++)
		{
			u &= v;
			fill(n + all(u), 0);
			b[i] = u;
		}
		u %= n; u[0] = 0;
		for (i = 0; i < l; i++) for (j = 1; j <= l; j++) if (i * l + j < n)
		{
			m = i * l + j - 1;
			ull r = 0, *f = b[i].data(), *g = a[j].data();
			for (k = 0; k <= m; k++) r = (r + f[k] * g[m - k]) % p;
			u[m + 1] = r * inv[m + 1] % p;
		}
		return u;
	}//8000 200ms
	Q shift(Q f, ull c)//get f(x+c),c\in [0,p)
	{
		int n = f.size(), i, j;
		Q g(n);
		for (i = 0; i < n; i++) (f[i] *= fac[i]) %= p;
		g[0] = 1;
		for (i = 1; i < n; i++) g[i] = g[i - 1] * c % p;
		for (i = 0; i < n; i++) (g[i] *= ifac[i]) %= p;
		f ^= g;
		for (i = 0; i < n; i++) (f[i] *= ifac[i]) %= p;
		return f;
	}//5e5 200ms (1 2 3 4 5) 3 -> (547 668 309 64 5)
	vector<ull> shift(vector<ull> y, ull c, ull m)//[0,n) 点值 -> [c,c+m) 点值
	{
		assert(y.size());
		if (y.size() == 1) return vector(m, y[0]);
		vector<ull> r, res;
		r.reserve(m);
		int n = y.size(), i, j, mm = m;
		while (c < n && m) r.push_back(y[c++]), --m;
		if (c + m > p)
		{
			res = shift(y, 0, c + m - p);
			m = p - c;
		}
		if (!m) { r.insert(r.end(), all(res)); return r; }
		int len = cal(m + n - 1), l = m + n - 1;
		for (i = n & 1; i < n; i += 2) y[i] = (p - y[i]) % p;
		for (i = 0; i < n; i++) y[i] = y[i] * ifac[i] % p * ifac[n - 1 - i] % p;
		y.resize(len);
		Q f, g;
		vector<ull> v(m + n - 1);
		c -= n - 1;
		for (i = 0; i < l; i++) v[i] = (c + i) % p;
		f = Q(y); g = Q(getinvs(v)) % len;
		f *= g;
		vector<ull> u(m);
		for (i = n - 1; i < l; i++) u[i - (n - 1)] = f[i];
		v.resize(m);
		for (i = 0; i < m; i++) v[i] = c + i;
		v = getinvs(v); c += n;
		ull tmp = 1;
		for (i = c - n; i < c; i++) tmp = tmp * i % p;
		for (i = 0; i < m; i++) (u[i] *= tmp) %= p, tmp = tmp * (c + i) % p * v[i] % p;
		r.insert(r.end(), all(u));
		r.insert(r.end(), all(res));
		assert(r.size() == mm);
		return r;
	}//5e5 430ms, (1 4 9 16) 3 5 -> (16 25 36 49 64)
	vector<ull> czt(Q f, ull c, int m)//求 f(c^[0,m))。核心 ij=C(i+j,2)-C(i,2)-C(j,2)
	{
		const static ull B = 1e5;
		static ull a[B + 2], b[B + 2];
		int i, n = f.size();
		if (m == 0) return { };
		if (c == 0)
		{
			vector<ull> r(m, f.fx(0));
			r[0] = f.fx(1);
			return r;
		}
		if (min(n, m) <= 16)
		{
			vector<ull> r(m);
			ull j;
			for (i = 0, j = 1; i < m; i++) r[i] = f.fx(j), j = j * c % p;
			return r;
		}
		ull l = cal(m += n - 1);
		ui cur = min<ui>(B, max({m, 4}));
		auto mic = [&](ull x) { return a[x % cur] * b[x / cur] % p; };
		Q g(l);
		assert(B * B > p);
		a[0] = b[0] = g[0] = g[1] = 1;
		for (i = 1; i <= cur; i++) a[i] = a[i - 1] * c % p;
		for (i = 1; i <= cur; i++) b[i] = b[i - 1] * a[cur] % p;
		for (i = 2; i < m; i++) g[i] = mic(i * (i - 1llu) / 2 % (p - 1));
		vector<ull> ig(m, 1);
		for (i = 2; i < m; i++) ig[i] = ig[i - 1] * g[i] % p;
		ull iv = ksm(ig[m - 1], p - 2);
		for (i = m - 1; i >= 2; i--)
		{
			ig[i] = ig[i - 1] * iv % p;
			if (i < n)
				(f[i] *= ig[i]) %= p;
			(iv *= g[i]) %= p;
		}
		reverse(all(f)); (f %= l) &= g;
		vector<ull> r(f.begin() + n - 1, f.begin() + m); m -= n - 1;
		for (i = 2; i < m; i++) (r[i] *= ig[i]) %= p;

		return r;
	}//luogu 1e6 500ms

	vector<ull> dft(const vector<ull> &a)
	{
		int n = a.size();
		if (n <= 1) return a;
		assert((p - 1) % n == 0);
		if ((n & -n) == n)
		{
			Q q(a, 0);
			q.dft();
			return vector<ull>(q.begin(), q.begin() + n);
		}
		ull wn = ksm(g, (p - 1) / n);
		return czt(Q(a), wn, n);
	}

	vector<ull> idft(const vector<ull> &a)
	{
		int n = a.size();
		if (n <= 1) return a;
		assert((p - 1) % n == 0);

		ull inv_wn = ksm(g, p - 1 - (p - 1) / n);
		vector<ull> res = czt(Q(a), inv_wn, n);

		ull inv_n = ksm(n, p - 2);
		for (ull &x : res) x = x * inv_n % p;
		return res;
	}
	vector<ull> Bell(int n)//B(0...n)
	{
		++n;
		Q f(n);
		int i;
		for (i = 1; i < n; i++) f[i] = ifac[i];
		f = exp_new(f);
		for (i = 2; i < n; i++) f[i] = f[i] * fac[i] % p;
		return vector<ull>(f.begin(), f.begin() + n);
	}
	vector<ull> S1_row(int n, int m)//S1(n,0...m),O(nlogn),unsigned
	{
		int cm = cal(++m);
		if (n == 0)
		{
			vector<ull> r(m);
			r[0] = 1;
			return r;
		}
		function<Q(int)> dfs = [&](int n) {
			if (n == 1)
			{
				Q f(2);
				f[1] = 1;
				return f;
			}
			Q f = dfs(n / 2);
			f *= shift(f, n / 2);
			if (n & 1)
			{
				f %= cal(n + 1);
				for (int i = n; i; i--) f[i] = f[i - 1];
				// for (int i=1; i<=n; i++) f[i]=f[i-1];
				--n;
				for (int i = 0; i <= n; i++) f[i] = (f[i] + f[i + 1] * n) % p;
			}
			if (f.size() > cm) f %= cm;
			return f;
		};
		Q f = dfs(n);
		if (f.size() < cm) f %= cm;
		return vector<ull>(f.begin(), f.begin() + m);
	}
	vector<ull> S1_column(int n, int m)//S1(0...n,m),O(nlogn)
	{
		if (m == 0)
		{
			vector<ull> r(n + 1);
			r[0] = 1;
			return r;
		}
		Q f(n + 1);
		int i;
		for (i = 1; i <= n; i++) f[i] = inv[i];
		f = pow(f, m);
		for (i = m; i <= n; i++) f[i] = f[i] * fac[i] % p * ifac[m] % p;
		return vector<ull>(f.begin(), f.begin() + n + 1);
	}
	vector<ull> S2_row(int n, int m)//S2(n,0...m),O(mlogm)
	{
		if (m == 0) return {n == 0};  // by codex
		int tm = ++m, i, j, cnt = 0;
		if (n == 0)
		{
			vector<ull> r(m);
			r[0] = 1;
			return r;
		}
		m = min(m, n + 1);
		vector<ull> pr(m), pw(m);
		pw[1] = 1;
		for (i = 2; i < m; i++)
		{
			if (!pw[i]) pr[cnt++] = i, pw[i] = ksm(i, n);
			for (j = 0; i * pr[j] < m; j++)
			{
				pw[i * pr[j]] = pw[i] * pw[pr[j]] % p;
				if (i % pr[j] == 0) break;
			}
		}
		Q f(m), g(m);
		for (i = 0; i < m; i += 2) f[i] = ifac[i];
		for (i = 1; i < m; i += 2) f[i] = p - ifac[i];
		// for (i=1; i<m; i++) g[i]=pw[i]*ifac[i]%p;
		for (i = 1; i < m; i++) g[i] = ksm(i, n) * ifac[i] % p;
		f *= g;
		vector<ull> r(f.begin(), f.begin() + m);
		r.resize(tm);
		return r;
	}//5e5 150ms
	vector<ull> S2_column(int n, int m)//S2(0...n,m),O(nlogn)
	{
		if (m == 0)
		{
			vector<ull> r(n + 1);
			r[0] = 1;
			return r;
		}
		Q f(n + 1);
		int i;
		for (i = 1; i <= n; i++) f[i] = ifac[i];
		f = pow(f, m);
		for (i = m; i <= n; i++) f[i] = f[i] * fac[i] % p * ifac[m] % p;
		return vector<ull>(f.begin(), f.begin() + n + 1);
	}//5e5 640ms
	vector<ull> signed_S1_row(int n, int m)
	{
		auto v = S1_row(n, m);
		for (int i = 1 ^ n & 1; i <= m; i += 2) v[i] = (p - v[i]) % p;
		return v;
	}//5e5 190ms
	vector<ull> Bernoulli(int n)//B(0...n)
	{
		int i;
		Q f(++n);
		for (i = 0; i < n; i++) f[i] = ifac[i + 1];
		f = ~f;
		for (i = 0; i < n; i++) f[i] = f[i] * fac[i] % p;
		return vector<ull>(f.begin(), f.begin() + n);
	}//5e5 180ms
	vector<ull> Partition(int n)//P(0...n)，拆分数
	{
		Q f(++n);
		int i, l = 0, r = 0;
		while (--l) if (3 * l * l - l >= n * 2) break;
		while (++r) if (3 * r * r - r >= n * 2) break;
		++l;
		for (i = l + abs(l) % 2; i < r; i += 2) f[3 * i * i - i >> 1] = 1;
		for (i = l + abs(l + 1) % 2; i < r; i += 2) f[3 * i * i - i >> 1] = p - 1;
		f = ~f;
		return vector<ull>(f.begin(), f.begin() + n);
	}//5e5 150ms
	struct reg
	{
		Q a00, a01, a10, a11;
		reg operator*(const reg &o) const
		{
			return {
				shrink(a00 * o.a00 + a01 * o.a10),
				shrink(a00 * o.a01 + a01 * o.a11),
				shrink(a10 * o.a00 + a11 * o.a10),
				shrink(a10 * o.a01 + a11 * o.a11)};
		}
		pair<Q, Q> operator*(const pair<Q, Q> &o) const
		{
			const auto &[b0, b1] = o;
			return {shrink(a00 * b0 + a01 * b1), shrink(a10 * b0 + a11 * b1)};
		}
	} E = {{vector{1llu}}, Q(), Q(), {vector{1llu}}};
	ostream &operator<<(ostream &cout, const reg &o)
	{
		return cout << "[" << o.a00 << ", " << o.a01 << "]\n"
			<< "[" << o.a10 << ", " << o.a11 << "]\n";
	}
	reg hgcd(Q a, Q b)
	{
		int m = a.deg() + 1 >> 1;
		if (m == 0) return E; // by codex
		if (b.deg() < m) return E;
		reg r = hgcd(a >> m, b >> m);
		auto [c, d] = r * pair{a, b};
		if (d.deg() < m) return r;
		auto [q, e] = div_mod(c, d);
		r.a00 -= shrink(q * r.a10);
		r.a01 -= shrink(q * r.a11);
		swap(r.a00, r.a10);
		swap(r.a01, r.a11);
		if (e.deg() < m) return r;
		int k = 2 * m - d.deg();
		auto s = hgcd(d >> k, e >> k);
		return s * r;
	}
	Q gcd(Q a, Q b)
	{
		if (a.deg() < b.deg()) swap(a, b);
		while (b.deg() >= 0)
		{
			a = mod(a, b);
			swap(a, b);
			auto tmp = hgcd(a, b);
			tie(a, b) = tmp * pair{a, b};
		}
		if (a.deg() == -1) return a;
		ull k = ksm(a[a.deg()], p - 2);
		for (int i = 0; i < a.size(); i++) a[i] = a[i] * k % p;
		return a;
	}
	vector<ull> root(Q f)
	{
		Q x(2);
		x[1] = 1;
		x = pow(x, p, f);
		if (x.size() < 2) x %= 2;
		(x[1] += p - 1) %= p;
		f = gcd(f, x);
		vector<ull> res;
		static mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
		function<void(Q)> dfs = [&](Q f) {
			int n = f.deg(), i;
			if (n <= 0) return;
			if (n == 1)
			{
				res.push_back((p - f[0]) % p);
				return;
			}
			Q g(n);
			for (i = 0; i < n; i++) g[i] = rnd() % p;
			g = gcd(pow(g, (p - 1) / 2, f) - 1, f);
			dfs(g); dfs(div(f, g));
		};
		dfs(f);
		sort(all(res));
		assert(unique(all(res)) == res.end());
		return res;
	}//4000 950ms
	optional<Q> inverse(Q a, Q m)
	{
		Q b = m;
		vector<pair<reg, Q>> buf;
		a = mod(a, b);
		swap(a, b);
		while (b.deg() >= 0)
		{
			auto [q, r] = div_mod(a, b);
			swap(a, r); swap(a, b);
			auto tmp = hgcd(a, b);
			tie(a, b) = tmp * pair{a, b};
			buf.emplace_back(move(tmp), q);
		}
		if (a.deg()) return { };
		reg res = E;
		reverse(all(buf));
		for (const auto &[tmp, q] : buf)
		{
			res = res * tmp;
			res.a00 -= shrink(q * res.a01);
			res.a10 -= shrink(q * res.a11);
			swap(res.a00, res.a01);
			swap(res.a10, res.a11);
		}
		return {res.a01 * ksm(a[0], p - 2)};
	}//5e4 950ms
	vector<ull> poly_to_newton(Q f, const vector<ull> &X)//f=sum(r_i*prod_{j<i}(x-x_j))
	{
		int n = X.size();
		if (n == 0) return { };
		assert(f.deg() < n);
		vector<ull> res(n);
		vector<Q> sum(4 * n + 4);
		function<void(int, int, int)> build = [&](int x, int l, int r) {
			if (l == r)
			{
				if (r != n - 1) sum[x] = Q(vector{(p - X[l]) % p, 1llu});
				return;
			}
			int mid = l + r >> 1, c = x * 2;
			build(c, l, mid); build(c + 1, mid + 1, r);
			sum[x] = shrink(sum[c] * sum[c + 1]);
		};
		build(1, 0, n - 1);
		function<void(Q &, int, int, int)> dfs = [&](Q &f, int x, int l, int r) {
			if (l == r)
			{
				res[l] = f[0];
				return;
			}
			if (l + 1 == r)
			{
				res[r] = f[1];
				res[l] = (f[0] + f[1] * X[l]) % p;
				return;
			}
			int c = x * 2, m = l + r >> 1;
			{
				Q g;
				tie(g, f) = div_mod(f, sum[c]);
				dfs(g, c | 1, m + 1, r);
			}
			dfs(f, c, l, m);
		};
		dfs(f, 1, 0, n - 1);
		return res;
	}//131072 559 ms
	vector<ull> poly_to_facpow(const Q &f, int n)
	{
		vector<ull> x(n);
		iota(all(x), 0);
		return poly_to_newton(f, x);
	}//luogu 1e5 368 ms
	Q newton_to_poly(const vector<ull> &g, const vector<ull> &X)
	{
		int n = g.size();
		if (n == 0) return Q();
		assert(n == X.size());
		vector<Q> sum(4 * n + 4);
		Q f(n);
		function<void(int, int, int)> build = [&](int x, int l, int r) {
			if (l == r)
			{
				if (r != n - 1) sum[x] = Q(vector{(p - X[l]) % p, 1llu});
				return;
			}
			int mid = l + r >> 1, c = x * 2;
			build(c, l, mid); build(c + 1, mid + 1, r);
			sum[x] = shrink(sum[c] * sum[c + 1]);
		};
		build(1, 0, n - 1);
		function<Q(int, int, int)> dfs = [&](int x, int l, int r) {
			if (l == r) return Q(vector<ull>{g[l]});
			if (l + 1 == r) return Q(vector<ull>{g[l] + g[r] * (p - X[l]) % p, g[r]});
			int c = x * 2, m = l + r >> 1;
			return (sum[c] *= dfs(c + 1, m + 1, r)) += dfs(c, l, m);
		};
		return dfs(1, 0, n - 1);
	}
	Q facpow_to_poly(const vector<ull> &g)
	{
		vector<ull> x(g);
		iota(all(x), 0);
		return newton_to_poly(g, x);
	}//luogu 2e5 449 ms
	vector<ull> mul_facpow(const vector<ull> &a, const vector<ull> &b)
	{
		int m = a.size() + b.size() - 1, n = cal(m * 2), i;
		Q f(a), g(b), h(n);
		f %= n, g %= n;
		for (i = 0; i < m; i++) h[i] = ifac[i];
		h.dft();
		f *= h, g *= h;
		for (i = 0; i < m; i++) (f[i] *= g[i] * fac[i] % p) %= p;
		fill(m + all(f), 0);
		rotate(h.begin(), n / 2 + all(h));
		f *= h;
		vector<ull> r(m);
		for (i = 0; i < m; i++) r[i] = f[i];
		return r;
	}//luogu 1e5 131 ms
}
using NTT::p;
using poly = NTT::Q;
