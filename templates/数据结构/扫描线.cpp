using T = ll;
vector<T> fun(vector<tuple<T, T, T, T>> &a)
{
	vector<T> x;
	for (auto [x1, y1, x2, y2] : a)
	{
		x.push_back(x1);
		x.push_back(x2);
	}
	sort(all(x)); x.resize(unique(all(x)) - x.begin());
	for (auto &[x1, y1, x2, y2] : a)
	{
		x1 = lower_bound(all(x), x1) - x.begin();
		x2 = lower_bound(all(x), x2) - x.begin();
	}
	return x;
}
struct sgt
{
	int n, z, y, d;
	vector<T> cnt, &p;
	vector<int> mn, lz;
	void build(int x, int l, int r)
	{
		cnt[x] = p[min(r, n - 1)] - p[l];
		if (l + 1 == r) return;
		int c = x * 2, m = l + r >> 1;
		build(c, l, m); build(c + 1, m, r);
	}
	sgt(vector<T> &p) :n(p.size()), p(p), cnt(n * 4), mn(n * 4), lz(n * 4) { build(1, 0, n); }
	void dfs(int x, int l, int r)
	{
		if (z <= l && r <= y)
		{
			mn[x] += d;
			lz[x] += d;
			return;
		}
		int c = x * 2, m = l + r >> 1;
		if (lz[x])
		{
			lz[c] += lz[x]; lz[c + 1] += lz[x];
			mn[c] += lz[x]; mn[c + 1] += lz[x];
			lz[x] = 0;
		}
		if (z < m) dfs(c, l, m);
		if (m < y) dfs(c + 1, m, r);
		mn[x] = min(mn[c], mn[c + 1]);
		cnt[x] = cnt[c] * (mn[x] == mn[c]) + cnt[c + 1] * (mn[x] == mn[c + 1]);
	}
	void modify(int l, int r, int dt)
	{
		z = l;
		y = r;
		d = dt;
		dfs(1, 0, n);
	}
};
T area(vector<tuple<T, T, T, T>> a)//[x1,y1,x2,y2], x1<y1, x2<y2
{
	int n = a.size(), i;
	auto X = fun(a);
	vector<tuple<T, int, T, T>> b(n * 2);
	for (i = 0; i < n; i++)
	{
		auto [x1, y1, x2, y2] = a[i];
		b[i] = {y1, -1, x1, x2};
		b[i + n] = {y2, 1, x1, x2};
	}
	sort(all(b), greater<>());
	sgt s(X);
	T lst = 0, ans = 0;
	for (auto [y, d, l, r] : b)
	{
		ans += (lst - y) * (X.back() - X[0] - s.cnt[1]);
		s.modify(l, r, d);
		lst = y;
	}
	return ans;
}
T perimeter_x(vector<tuple<T, T, T, T>> a)
{
	int n = a.size(), i;
	auto X = fun(a);
	vector<tuple<T, int, T, T>> b(n * 2);
	for (i = 0; i < n; i++)
	{
		auto [x1, y1, x2, y2] = a[i];
		b[i] = {y1, -1, x1, x2};
		b[i + n] = {y2, 1, x1, x2};
	}
	sort(all(b), greater<>());
	sgt s(X);
	T lst = s.cnt[1], ans = 0;
	for (auto [y, d, l, r] : b)
	{
		s.modify(l, r, d);
		T cur = s.cnt[1];
		ans += abs(lst - cur);
		lst = cur;
	}
	return ans;
}
T perimeter(vector<tuple<T, T, T, T>> a)//[x1,y1,x2,y2], x1<y1, x2<y2
{
	T ansx = perimeter_x(a);
	for (auto &[x1, y1, x2, y2] : a)
	{
		swap(x1, y1);
		swap(x2, y2);
	}
	T ansy = perimeter_x(a);
	return ansx + ansy;
}
