namespace Gauss
{
	typedef double db;
	const db eps = 1e-8;
	template<class T> pair<vector<db>, int> solve(const vector<vector<T>> &A)//和为 0。返回秩，负数无解
	{
		assert(A.size());
		int n = A.size(), m = A[0].size() - 1, i, j, k, l, r, fg = 1;
		db a[n][m + 1], b;
		vector<int> w;
		for (i = 0; i < n; i++) for (j = 0; j <= m; j++) a[i][j] = A[i][j];
		for (i = l = r = 0; i < n && l < m; i++, l++)
		{
			k = i;
			for (j = i + 1; j < n; j++) if (fabs(a[j][l]) > fabs(a[k][l])) k = j;
			if (fabs(a[k][l]) < eps) { --i; continue; }
			if (i != k) for (j = l; j <= m; j++) swap(a[i][j], a[k][j]);
			w.push_back(l);
			b = 1 / a[i][l]; ++r; a[i][l] = 1;
			for (j = l + 1; j <= m; j++) a[i][j] *= b;
			for (j = 0; j < n; j++) if (i != j)
			{
				b = a[j][l]; a[j][l] = 0;
				for (k = l + 1; k <= m; k++) a[j][k] -= b * a[i][k];
			}
		}
		vector<db> X(m);
		for (j = 0; j < w.size(); j++) X[w[j]] = -a[j][m];
		for (j = 0; j < n && ~fg; j++)
		{
			b = a[j][m];
			for (k = 0; k < m; k++) b += X[k] * a[j][k];
			if (fabs(b) > eps) fg = -1;
		}
		return {X, r * fg};
	}
}
//?
