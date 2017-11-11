#!/usr/bin/perl

use strict;


my @todo;

foreach my $src (qw(it pt ko ru en es de fr ja)) {
    foreach my $tgt (qw(it pt ko ru en es de fr ja)) {
        push @todo,[$src,$tgt, 123]
    }
}
foreach my $src (qw(it pt ko ru en es de fr ja)) {
    foreach my $tgt (qw(it pt ko ru en es de fr ja)) {
        push @todo,[$src,$tgt, 768]
    }
}


sub countslot {
  my @r=split(/\n/,`nvidia-smi | grep python`);
  return $#r+1;
}

sub dolaunch {
    my ($cmd)=@_;
    print("RUN CMD: $cmd\n");
    system($cmd);
}

while(my $lp=shift @todo) {
    while(countslot()>=9) { sleep(30); }
    my @lp=@{$lp};
    print "************* LAUNCH $lp[0]-$lp[1]-$lp[2]\n";
    dolaunch("scripts/run-cos.sh $lp[0] $lp[1] $lp[2] 2> /dev/null &");
    sleep(30);
}
