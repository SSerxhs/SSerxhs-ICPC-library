struct Q
{
	int x,y;
	Q operator+(const Q &o) const {return {x+o.x,y+o.y};}
	Q operator-(const Q &o) const {return {x-o.x,y-o.y};}
	Q & operator+=(const Q &o) {x+=o.x;y+=o.y;return *this;}
	Q & operator-=(const Q &o) {x-=o.x;y-=o.y;return *this;}
	bool operator<(const Q &o) const {return x==o.x?y<o.y:x<o.x;}
	bool operator==(const Q &o) const {return x==o.x&&y==o.y;}
	bool operator!=(const Q &o) const {return x!=o.x||y!=o.y;}
	ll len() const {return (ll)x*x+(ll)y*y;}
};
const Q d[4]={{0,1},{1,0},{0,-1},{-1,0}};
//const Q d[8]={{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}};

	auto av=[&](const Q &o) -> bool
	{
		return o.x>=0&&o.y>=0&&o.x<n&&o.y<m;
	};
	auto av=[&](const int &x,const int &y) -> bool
	{
		return x>=0&&y>=0&&x<n&&y<m;
	};
bool av(const int &x,const int &y)
{
	return x>=0&&y>=0&&x<n&&y<m;
}
const int dx[4]={0,1,0,-1},dy[4]={1,0,-1,0};
//const int dx[8]={0,1,1,1,0,-1,-1,-1},dy[8]={1,1,0,-1,-1,-1,0,1};