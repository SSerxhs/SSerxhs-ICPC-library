template<typename T> struct GCD
{
	vector<pair<int,T>> res;
	GCD(const vector<T> &a):res(n)
	{
		int n=a.size(),i,j;
		vector<ll> v(n);
		vector<int> l(n);
		vector<vector<pair<int,T>> res(n);
		for (i=0; i<n; i++)
		{
			for (v[i]=a[i],j=l[i]=i; j>=0; j=l[j]-1)
			{
				v[j]=gcd(v[j],a[i]);
				while (l[j]&&gcd(a[i],v[l[j]-1])==gcd(a[i],v[j])) l[j]=l[l[j]-1];
				//[l[j]..j,i]区间内的值求fun均为v[j]
			}
			for (j=i; j>=0; j=l[j]) res[i].push_back({l[j],v[j]});
			reverse(all(res[i]));
		}
	}
	T ask(int l,int r)//[l,r]
	{
		return res[r].prev(upper_bound(l))->second;
	}
};