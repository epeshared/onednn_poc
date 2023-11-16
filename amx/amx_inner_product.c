#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>

#define AB(x) if(x){fprintf(stderr, "%s:%d ABORT, %s\n", __FILE__, __LINE__, #x); exit(-__LINE__);}
#define TMM_MAX_CAP 1024
#define SET_TILE(idx, rows, bpr) {short* bytePerRow = &(cfg[16+2*(idx)]); *bytePerRow=(bpr); cfg[48+(idx)]=(rows);}
#define UPDATE_CFG _tile_loadconfig(cfg);
#define LOAD(idx, ptr, stride) _tile_loadd((idx), (ptr), (stride))
#define MM(C_, A_, B_) _tile_dpbssd(C_, A_, B_)
#define SV(idx_, ptr_, stride_) _tile_stored(idx_, ptr_, stride_)
#define ZERO(idx) _tile_zero(idx)
enum{
    t0 = 0,
    t1, t2, t3, t4, t5, t6, t7
};
typedef unsigned char u8;
typedef unsigned short u16;
#if defined(__linux)
#include <unistd.h>
#include <sys/syscall.h>
#define DIE_IF(x) {long rc = (x); if(rc){/*printf("%s failed: %l\n", #x, rc); */return rc;}}
#define XFEATURE_XTILECFG	17
#define XFEATURE_XTILEDATA	18
#define XFEATURE_MASK_XTILECFG	(1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA	(1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE	(XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM	0x1022
#define ARCH_REQ_XCOMP_PERM	0x1023
static int req_perm_xtile() {
  unsigned long bitmask;
  DIE_IF(syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA));
  DIE_IF(syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask));
  if (bitmask & XFEATURE_MASK_XTILE) return 0;
  return -1;
}
#endif
int amx(const char* aVec, const char* bVec, size_t byteDim){
    u8 cfg[64];
    size_t i, intDim = byteDim >> 2;
    int c;
    AB(63 & byteDim);
    memset(cfg, 0, 64);
    cfg[0] = 1; //an intialized 

    //这里我们只用tile 0(aVec), tile 1(bVec), tile 2(结果)
    SET_TILE(0, 1, 64);	    //t0 1 x 64 byte
    SET_TILE(1, 16, 4);	    //t1 16 x 4  byte
    SET_TILE(2, 1, 4);		    //t2 1 x 1  int
    UPDATE_CFG;
    

    ZERO(2);
    for(i = 0; i < byteDim; i+=64){
        _tile_loadd(0, &(aVec[i]), 4);
        _tile_loadd(1, &(bVec[i]), 4);
        /*
        LOAD(0, &(aVec[i]), 4);
        LOAD(1, &(bVec[i]), 4);
        MM(2, 0, 1);
        */
        _tile_dpbssd(2, 0, 1);
    }
    SV(2, &c, 0);
    return c;
}
int amxRef(const char* aVec, const char* bVec, size_t dim){
    size_t k;
    int sum = 0;
    for(k = 0; k < dim; ++k){
	sum += aVec[k] * ((int) bVec[k]);
    }
    return sum;
}


char* createVec(size_t col){
    char* ptr = NULL;
    size_t i, j;
    AB(64 & col);
    
    ptr = aligned_alloc(64, col);
    AB(NULL == ptr);

    srand(time(NULL));
    for(i = 0; i < col; ++i){
	ptr[i] = (char)( (rand() & 255) - 128);
    }
    return ptr;
}

int main(int argc, char** argv){
    const size_t dim  = 512; 

    char* aVec = createVec(dim);
    AB(NULL == aVec);
    char* bVec = createVec(dim);
    AB(NULL == bVec);

    AB(0 != req_perm_xtile());//在这台机器上需要初始化, 否则会illegal instruction

    int c,d;
    struct timeval start, end;
    long seconds, microseconds;
    double elapsed;  

    gettimeofday(&start, NULL);
    for (int i = 0; i < 1000000; i++) {
        c = amxRef(aVec, bVec, dim);
    }
    gettimeofday(&end, NULL);
    printf("\n");
    seconds = end.tv_sec - start.tv_sec;
    microseconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + microseconds*1e-6;
    printf("Function scalar took %f seconds to execute.\n", elapsed); 

    gettimeofday(&start, NULL);
    for (int i = 0;i < 1000000; i++) {
        d = amx(aVec, bVec, dim);
    }
    gettimeofday(&end, NULL);
    printf("\n");
    seconds = end.tv_sec - start.tv_sec;
    microseconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + microseconds*1e-6;
    printf("Function amx took %f seconds to execute.\n", elapsed); 

    // int c = amxRef(aVec, bVec, dim);
    // int d = amx(aVec, bVec, dim);
    if(c != d){
	fprintf(stdout, "\033[41mWrong ref:%d != %d:amx\033[0m\n", c, d);
    }else{
	fprintf(stdout, "\033[42mCorrect\033[0m\n");
    }

    free(aVec);
    free(bVec);
}
