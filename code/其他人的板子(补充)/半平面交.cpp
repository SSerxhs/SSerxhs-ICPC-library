const int N=305;
const db inf=1e15,eps=1e-10;
int sign(db x){
	if(fabs(x)<eps)return 0;
	return x>0?1:-1;
}

struct vec{
	db x,y;
	vec(){}
	vec(db a,db b){x=a,y=b;}
	vec operator+(const vec& p)const{
		return vec(x+p.x,y+p.y);
	}
	vec operator-(const vec& p)const{
		return vec(x-p.x,y-p.y);
	}
	db operator*(const vec& p)const{
		return x*p.y-y*p.x;
	}
	vec operator*(const db& p)const{
		return vec(x*p,y*p);
	}
}p1[N],p2[N];

struct line{
	vec s,t;
	line(){}
	line(vec a,vec b){s=a,t=b;}
}a[N],q[N];
db ang(vec v){
	return atan2(v.y,v.x);
}
db ang(line l){
	return ang(l.t-l.s);
}
bool cmp(line x,line y){
	int s=sign(ang(x)-ang(y));
	return s?s<0:sign((x.t-x.s)*(y.t-x.s))>0;
}

vec inter(line x,line y){
	vec a=y.s-x.s,b=x.t-x.s,c=y.t-y.s;
	return y.s+c*((a*b)/(b*c));
}
bool out(line l,vec p){
	return sign((l.t-l.s)*(p-l.s))<0;
}

int n,tot=0;
db ans=inf;
int main(){
	scanf("%d",&n);
	for(int i=1;i<=n;++i)scanf("%lf",&p1[i].x);
	for(int i=1;i<=n;++i)scanf("%lf",&p1[i].y);
	for(int i=1;i<n;++i)a[i]=line(p1[i],p1[i+1]);
	a[n]=line(vec(p1[1].x,inf),vec(p1[1].x,p1[1].y));
	a[n+1]=line(vec(p1[n].x,p1[n].y),vec(p1[n].x,inf));
	
	sort(a+1,a+n+2,cmp);
	for(int i=1;i<=n;++i){
		if(!sign(ang(a[i])-ang(a[i+1])))continue;
		a[++tot]=a[i];
	}a[++tot]=a[n+1];
	
	int l=1,r=0;
	q[++r]=a[1],q[++r]=a[2];
	for(int i=3;i<=tot;++i){
		while(l<r&&out(a[i],inter(q[r],q[r-1])))--r;
		while(l<r&&out(a[i],inter(q[l],q[l+1])))++l;
		q[++r]=a[i];
	}
	while(l<r&&out(q[l],inter(q[r],q[r-1])))--r;
	while(l<r&&out(q[r],inter(q[l],q[l+1])))++l;
//......
}
