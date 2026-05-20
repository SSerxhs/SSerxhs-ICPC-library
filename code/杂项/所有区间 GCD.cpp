template<class T> struct GCD
{
	vector<pair<int, T>> res;
	GCD(const vector<T> &a) :res(n)
	{
		int n = a.size(), i, j;
		vector<ll> v(n);
		vector<int> l(n);
		for (i = 0; i < n; i++)
		{
			for (v[i] = a[i], j = l[i] = i; j >= 0; j = l[j] - 1)
			{
				v[j] = fun(v[j], a[i]);
				while (l[j] && fun(a[i], v[l[j] - 1]) == fun(a[i], v[j])) l[j] = l[l[j] - 1];
				//[l[j]..j,i]区间内的值求 fun 均为 v[j]
			}
		}
	}
};
