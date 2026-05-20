namespace FFT
{
#define all(x) (x).begin(),(x).end()
	typedef double db;
	const int N = 1 << 21;
	const db pi = 3.14159265358979323846;
	struct comp
	{
		db x, y;
		comp operator+(const comp &o) const { return {x + o.x, y + o.y}; }
		comp operator-(const comp &o) const { return {x - o.x, y - o.y}; }
		comp operator*(const comp &o) const { return {x * o.x - y * o.y, o.x * y + x * o.y}; }
		comp operator*(const db &o) const { return {x * o, y * o}; }
		void operator*=(const comp &o) { *this = {x * o.x - y * o.y, o.x * y + x * o.y}; }
		void operator*=(const db &o) { x *= o; y *= o; }
		void operator/=(const db &o) { x /= o; y /= o; }
	};
	long long dtol(const double &x) { return fabs(round(x)); }
	const comp I{0, -1};
	ostream &operator<<(ostream &cout, const comp &o) { cout << o.x; if (o.y >= 0) cout << '+'; return cout << o.y << 'i'; }
	int r[N];
	char c;
	comp Wn[N];
	void init(int n)
	{
		static int preone = -1;
		if (n == preone) return;
		preone = n;
		int b, i;
		b = __builtin_ctz(n) - 1;
		for (i = 1; i < n; i++) r[i] = r[i >> 1] >> 1 | (i & 1) << b;
		for (i = 0; i < n; i++) Wn[i] = {cos(pi * i / n), sin(pi * i / n)};
	}
	int cal(int x) { return 1u << 32 - __builtin_clz(max(x, 2) - 1); }
	struct Q
	{
		vector<comp> a;
		int deg;
		comp *pt() { return a.data(); }
		Q(int n = 0)
		{
			deg = n;
			a.resize(cal(n));
		}
		void dft(int xs = 0)//1,0
		{
			int i, j, k, l, n = a.size(), d;
			comp w, wn, b, c, *f = pt(), *g, *a = f;
			init(n);
			if (xs) reverse(a + 1, a + n);//spe
			for (i = 0; i < n; i++) if (i < r[i]) swap(a[i], a[r[i]]);
			for (i = 1, d = 0; i < n; i = l, d++)
			{
				//wn={cos(pi/i),(xs?-1:1)*sin(pi/i)};
				l = i << 1;
				for (j = 0; j < n; j += l)
				{
					//w={1,0};
					f = a + j; g = f + i;
					for (k = 0; k < i; k++)
					{
						w = Wn[k * (n >> d)];
						b = f[k]; c = g[k] * w;
						f[k] = b + c;
						g[k] = b - c;
						//w*=wn;
					}
				}
			}
			if (xs) for (i = 0; i < n; i++) a[i] /= n;
		}
		void operator|=(Q o)
		{
			int n = deg + o.deg - 1, m = cal(n), i;
			a.resize(m); o.a.resize(m);
			dft(); o.dft();
			for (i = 0; i < m; i++) a[i] *= o.a[i];
			dft(1);
			for (i = n; i < m; i++) a[i] = { };
			deg = n;
		}
		Q operator|(Q o) const { o |= *this; return o; }
	};
	Q mul(Q a, const Q &b)//三次变两次，仅实数，注意精度
	{
		int n = a.deg + b.deg - 1, m = cal(n), i;
		a.a.resize(m);
		for (i = 0; i < b.deg; i++) a.a[i] = {a.a[i].x, b.a[i].x};
		a.dft();
		for (i = 0; i < m; i++) a.a[i] *= a.a[i];
		a.dft(1);
		for (i = 0; i < n; i++) a.a[i] = {a.a[i].y * .5};
		for (i = n; i < m; i++) a.a[i] = { };
		a.deg = n;
		return a;
	}
	void ddt(Q &a, Q &b)//double dft，仅实数，注意精度
	{
		comp x, y;
		int n = a.a.size(), i;
		assert(n == b.a.size());
		for (i = 0; i < n; i++) a.a[i] = {a.a[i].x, b.a[i].x};
		a.dft();
		for (i = 0; i < n; i++) b.a[i] = {a.a[i].x, -a.a[i].y};
		reverse(b.pt() + 1, b.pt() + n);
		for (i = 0; i < n; i++)
		{
			x = a.a[i]; y = b.a[i];
			a.a[i] = (x + y) * .5;
			b.a[i] = (y - x) * .5 * I;
		}
	}
}
using FFT::dtol;
