vector<ui> bm(const vector<ui> &a)
{
	vector<ui> r, lst;
	int n = a.size(), m = 0, q = 0, i, j, k = -1;
	ui D = 0;
	for (i = 0; i < n; i++)
	{
		ui cur = 0;
		for (j = 0; j < m; j++) cur = (cur + (ull)a[i - j - 1] * r[j]) % p;
		cur = (a[i] + p - cur) % p;
		if (!cur) continue;
		if (k == -1)
		{
			k = i;
			D = cur;
			r.resize(m = i + 1);
			continue;
		}
		auto v = r;
		ui x = (ull)cur * ksm(D, p - 2) % p;
		if (m < q + i - k) r.resize(m = q + i - k);
		(r[i - k - 1] += x) %= p;
		ui *b = r.data() + i - k;
		x = (p - x) % p;
		for (j = 0; j < q; j++) b[j] = (b[j] + (ull)x * lst[j]) % p;
		if (v.size() + k < lst.size() + i)
		{
			lst = v;
			q = v.size();
			k = i;
			D = cur;
		}
	}
	return r;
}
#define safe
struct Q
{
	int x, y;
	ui w;
};
mt19937_64 rnd(9980);
vector<ui> minpoly(int n, const vector<Q> &a)//[0,n),max:1
{
	for (auto [x, y, w] : a) assert(min(x, y) >= 0 && max(x, y) < n);
	vector<ui> u(n), v(n), b(n * 2 + 1), tmp(n);
	int i;
	for (ui &x : u) x = rnd() % p;
	for (ui &x : v) x = rnd() % p;
	assert(*min_element(all(u)) && *min_element(all(v)));
	for (ui &r : b)
	{
		for (i = 0; i < n; i++) r = (r + (ull)u[i] * v[i]) % p;
		fill(all(tmp), 0);
		for (auto [x, y, w] : a) tmp[x] = (tmp[x] + (ull)w * v[y]) % p;
		swap(v, tmp);
	}
	auto r = bm(b);
#ifdef safe
	for (ui &x : u) x = rnd() % p;
	for (ui &x : v) x = rnd() % p;
	for (ui &r : b)
	{
		for (i = 0; i < n; i++) r = (r + (ull)u[i] * v[i]) % p;
		fill(all(tmp), 0);
		for (auto [x, y, w] : a) tmp[x] = (tmp[x] + (ull)w * v[y]) % p;
		swap(v, tmp);
	}
	auto rr = bm(b);
	assert(r == rr);
#endif
	reverse(all(r));
	for (ui &x : r) if (x) x = p - x;
	r.push_back(1);
	return r;
}
ui det(int n, vector<Q> a)//[0,m)
{
	vector<ui> b(n);
	for (ui &x : b) x = rnd() % p;
	assert(*min_element(all(b)));
	for (auto &[x, y, w] : a) w = (ull)w * b[x] % p;
	ui r = minpoly(n, a)[0], tmp = 1;
	for (ui x : b) tmp = (ull)tmp * x % p;
	r = (ull)r * ksm(tmp, p - 2) % p;
#ifdef safe
	for (ui &x : b) x = rnd() % p;
	assert(*min_element(all(b)));
	for (auto &[x, y, w] : a) w = (ull)w * b[x] % p;
	ui rr = minpoly(n, a)[0], tmpp = 1;
	for (ui x : b) tmpp = (ull)tmpp * x % p;
	rr = (ull)rr * ksm(tmpp, p - 2) % p * ksm(tmp, p - 2) % p;
	assert(r == rr);
#endif
	return n & 1 ? (p - r) % p : r;
}
vector<ui> gauss(const vector<Q> &a, vector<ui> v)
{
	int n = v.size(), i, j;
	for (auto [x, y, w] : a) assert(0 <= x && x < n && 0 <= y && y < n);
	vector<ui> u(n), b(2 * n + 1), tmp(n), tv = v;
	for (ui &x : u) x = rnd() % p;
	assert(*min_element(all(u)));
	for (ui &r : b)
	{
		for (i = 0; i < n; i++) r = (r + (ull)u[i] * v[i]) % p;
		fill(all(tmp), 0);
		for (auto [x, y, w] : a) tmp[x] = (tmp[x] + (ull)w * v[y]) % p;
		swap(v, tmp);
	}
	auto f = bm(b);
	f.insert(f.begin(), p - 1);
	int m = (int)f.size() - 2;
	v = tv; fill(all(u), 0);
	ui x;
	for (i = 0; i <= m; i++)
	{
		x = f[m - i];
		for (j = 0; j < n; j++) u[j] = (u[j] + (ull)v[j] * x) % p;
		fill(all(tmp), 0);
		for (auto [x, y, w] : a) tmp[x] = (tmp[x] + (ull)w * v[y]) % p;
		swap(v, tmp);
	}
	x = ksm((p - f.back()) % p, p - 2);
	for (ui &y : u) y = (ull)y * x % p;
#ifdef safe
	for (auto [x, y, w] : a) tv[x] = (tv[x] + (ull)(p - w) * u[y]) % p;
	assert(!*min_element(all(tv)));
#endif
	return u;
}
