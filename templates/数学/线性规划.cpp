using db = long double;//__float128
struct linear
{
	static const int N = 45;//n+m
	db r[N][N];
	int col[N], row[N];
	const db eps = 1e-10, inf = 1e9;//1e-17
	int n, m;
	template<class T> linear(const vector<T> &a)//target: maximize \sum a(i-1)xi
	{
		memset(r, 0, sizeof r);
		memset(col, 0, sizeof col);
		memset(row, 0, sizeof row);
		n = a.size(); m = 0;
		for (int i = 1; i <= n; i++) r[0][i] = -a[i - 1];
	}
	template<class T> void add(const vector<T> &a, db b)//limit: \sum a(i-1)xi<=b
	{
		assert(a.size() == n);
		++m;
		for (int i = 1; i <= n; i++) r[m][i] = -a[i - 1];
		r[m][0] = b;
	}
	void pivot(int k, int t)
	{
		swap(row[k + n], row[t]);
		db rkt = -r[k][t];
		int i, j;
		for (i = 0; i <= n; i++) r[k][i] /= rkt;
		r[k][t] = -1 / rkt;
		for (i = 0; i <= m; i++) if (i != k)
		{
			db rit = r[i][t];
			if (rit >= -eps && rit <= eps) continue;
			for (j = 0; j <= n; j++) if (j != t) r[i][j] += rit * r[k][j];
			r[i][t] = r[k][t] * rit;
		}
	}
	bool init()
	{
		int i;
		for (i = 1; i <= n + m; i++) row[i] = i;
		while (1)
		{
			int q = 1;
			auto b_min = r[1][0];
			for (i = 2; i <= m; i++) if (r[i][0] < b_min) b_min = r[i][0], q = i;
			if (b_min + eps >= 0) return 1;
			int p = 0;
			for (i = 1; i <= n; i++) if (r[q][i] > eps && (!p || row[i] > row[p])) p = i;
			if (!p) break;
			pivot(q, p);
		}
		return 0;
	}
	bool simplex()
	{
		while (1)
		{
			int t = 1, k = 0, i;
			for (i = 2; i <= n; i++) if (r[0][i] < r[0][t]) t = i;
			if (r[0][t] >= -eps) return 1;
			db ratio_min = inf;
			for (i = 1; i <= m; i++) if (r[i][t] < -eps)
			{
				db ratio = -r[i][0] / r[i][t];
				if (!k || ratio<ratio_min || ratio <= ratio_min + eps && row[i]>row[k])
				{
					ratio_min = ratio;
					k = i;
				}
			}
			if (!k) break;
			pivot(k, t);
		}
		return 0;
	}
	void solve(int type)
	{
		if (!init())
		{
			cout << "Infeasible\n";
			return;
		}
		if (!simplex())
		{
			cout << "Unbounded\n";
			return;
		}
		cout << (long double)(-r[0][0]) << '\n';
		if (type)
		{
			int i;
			memset(col + 1, 0, n * sizeof col[0]);
			for (i = n + 1; i <= n + m; i++) col[row[i]] = i;
			for (i = 1; i <= n; i++) cout << (long double)(col[i] ? r[col[i] - n][0] : 0) << " \n"[i == n];
		}
	}
};
