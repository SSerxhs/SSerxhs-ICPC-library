template <class T> vector<T> min_plus_convolution(const vector<T> &a, const vector<T> &b)
{
	int n = a.size(), m = b.size(), i;
	vector<T> c(n + m - 1);
	function<void(int, int, int, int)> dfs = [&](int l, int r, int ql, int qr) {
		if (l > r) return;
		int mid = l + r >> 1;
		while (ql + m <= l) ++ql;
		while (qr > r) --qr;
		int qmid = -1;
		c[mid] = inf;
		for (int i = ql; i <= qr; i++) if (mid - i >= 0 && mid - i < m && cmin(c[mid], a[i] + b[mid - i])) qmid = i;
		dfs(l, mid - 1, ql, qmid);
		dfs(mid + 1, r, qmid, qr);
	};
	dfs(0, n + m - 2, 0, n - 1);
	return c;
}

