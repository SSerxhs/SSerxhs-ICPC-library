template<class T> void min_order(vector<T> &a)
{
	int n = a.size(), i, j, k;
	a.resize(n * 2);
	for (i = 0; i < n; i++) a[i + n] = a[i];
	i = k = 0; j = 1;
	while (i < n && j < n && k < n)
	{
		T x = a[i + k], y = a[j + k];
		if (x == y) ++k; else
		{
			(x > y ? i : j) += k + 1;
			j += (i == j);
			k = 0;
		}
	}
	a.resize(n);
	//[min(i,j),n)+[0,min(i,j))
	rotate(a.begin(), min(i, j) + all(a));
}
