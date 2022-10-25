vector<int> lg(2);
template <typename T> struct mintable
{
	vector<T> a;
	vector<vector<T>> st;
	mintable(const vector<T> &b):a(all(b))
	{
		int n=a.size(),i,j,k,r;
		while (lg.size()<=n) lg.push_back(lg[lg.size()>>1]+1);
		st.assign(lg[n]+1,vector<T>(n));
		iota(all(st[0]),0);
		for (j=1;j<=lg[n];j++)
		{
			r=n-(1<<j);
			k=1<<j-1;
			for (i=0;i<=r;i++) st[j][i]=a[st[j-1][i]]<a[st[j-1][i+k]]?st[j-1][i]:st[j-1][i+k];
		}
	}
	T rmq(int l,int r) const
	{
		assert(0<=l&&l<=r&&r<a.size());
		int z=lg[r-l+1];
		return min(a[st[z][l]],a[st[z][r-(1<<z)+1]]);
	}
	int rmp(int l,int r) const
	{
		assert(0<=l&&l<=r&&r<a.size());
		int z=lg[r-l+1];
		return a[st[z][l]]<a[st[z][r-(1<<z)+1]]?st[z][l]:st[z][r-(1<<z)+1];
	}
};