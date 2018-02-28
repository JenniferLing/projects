#!/usr/bin/perl -w
use strict;
use Data::Dumper;

my $numFolds = 4;
my $dom_path = "EMEA.psd";
my $seenFName = "seen.hansard.gz";
my $USAGE = "usage: get_folds.pl -dom <path to domain.psd> -k <k-fold cross validation> -seen <path to seen.hansards.gz file> -max <maximal number of tokens per type> -out <path to file with fold information>";
my $maxTokPerType = 100;
my $outname = "type_to_fold.info";

while (1) {
    my $arg = shift or last;
    if    ($arg eq '-dom') { $dom_path = shift or die "-dom needs an argument"; }
    elsif ($arg eq '-k') { $numFolds = shift or die "-k needs an argument"; }
    elsif ($arg eq '-seen')     { $seenFName = shift or die "-seen needs an argument"; }
	elsif ($arg eq '-max')     { $maxTokPerType = shift or die "-max needs an argument"; }
	elsif ($arg eq '-out')     { $outname = shift or die "-out needs an argument"; }
    else { die $USAGE; }
}

#print "Start perl script\n";

my %allDom = ();

$allDom{$dom_path} = 1;

my %seen_or_mfs = ();
%seen_or_mfs = readSeenList();

# print Dumper(\%seen_or_mfs) . "\n";

my %warnUnseen = ();
my %numTypes = ();

my @allData = ();
my $N = 0;  my $Np = 0; my $Nn = 0;
foreach my $dom (keys %allDom) {
    my @thisData = generateData($dom);
    # print Dumper(\@thisData) . "\n";
    for (my $i=0; $i<@thisData; $i++) {
        if ($thisData[$i]{'label'} eq '') { next; }

        %{$allData[$N]} = %{$thisData[$i]};
        $allData[$N]{'domain'} = $dom;
        if    ($allData[$N]{'label'} eq '') {}
        elsif ($allData[$N]{'label'} > 0  ) { $Np++; }
        else                                { $Nn++; }
        $N++;
    }
}

# print Dumper(\@allData) . "\n";

if ($N == 0) { die "did not read any data!"; }

doEvenSplit();

sub generateData {
    my ($dom) = @_;

    my @Y = (); my @W = ();
    open F, "$dom" or die $!;
#    open O, "> source_data/$dom.psd.markedup" or die $!;
    while (<F>) {
        chomp;
        my ($snt_id, $fr_start, $fr_end, $en_start, $en_end, $fr_phrase, $en_phrase) = split /\t/, $_;
        my $Y = '';

		if (not exists $seen_or_mfs{$fr_phrase}) {
			$warnUnseen{$fr_phrase} = 1;
		} else {
			$Y = (exists $seen_or_mfs{$fr_phrase}{$en_phrase}) ? -1 : 1;
		}
#        print O $Y . "\t" . $_ . "\n";

        push @W, $fr_phrase;
        push @Y, $Y;
    }
    close F;
#    close O;

	my %type = ();
    
    my @F = ();
    for (my $n=0; $n<@W; $n++) {
        %{$F[$n]} = ();
        $F[$n]{'label'} = $Y[$n];
        if ($Y[$n] eq '') { next; }
        $F[$n]{'phrase'} = $W[$n];
        if (exists $type{$W[$n]}) {
            foreach my $f (keys %{$type{$W[$n]}}) {
                $F[$n]{$f} = $type{$W[$n]}{$f};
            }
        }
        $F[$n]{'___bias'} = 1;
    }

    my @Fpruned = ();
    for (my $n=0; $n<@F; $n++) {
        my $p = $F[$n]{'phrase'};
        if (not defined $p) { next; }
        
        if ((defined $numTypes{$p}) && ($numTypes{$p} >= $maxTokPerType)) { next; }
        $numTypes{$p}++;

        my $n0 = scalar @Fpruned;
        foreach my $k (keys %{$F[$n]}) {
            $Fpruned[$n0]{$k} = $F[$n]{$k};
        }
    }

    return (@Fpruned);
}

sub readSeenList {
    open F, "zcat $seenFName|" or die $!;
    my %seen = ();
    while (<F>) {
        chomp;
        my ($fr_phrase, $en_phrase, $p_e_given_f) = split /\t/, $_;
        if (defined $p_e_given_f) {
            $seen{$fr_phrase}{$en_phrase} = $p_e_given_f;
        }
    }
    close F;

    return (%seen);
}

sub log0 {
    my ($v) = @_;
    if ($v <= 0) { return 0; }
    return log($v);
}

sub doEvenSplit {
    # replace numfolds with a power of 2
    my $oldNumFolds = $numFolds;
    my $logNumFolds = int(log($numFolds) / log(2));
    $numFolds = 2 ** $logNumFolds;
    if ($numFolds != $oldNumFolds) {
        print STDERR "warning: using $numFolds instead of $oldNumFolds (need a power of 2 for even splitting)\n";
        #if ($numFolds < 3) { die "cannot have fewer than 3 folds!!!"; }
    }
    
    # print STDERR "Num folds: $numFolds\n";

    my %typeSize = ();
    my %availableTypes = ();
    for (my $n=0; $n<$N; $n++) {
        my $type = $allData[$n]{'phrase'};
        $typeSize{ $type }{N} += ( $allData[$n]{'label'} > 0 ) ? 0 : 1;
        $typeSize{ $type }{P} += ( $allData[$n]{'label'} > 0 ) ? 1 : 0;
        $typeSize{ $type }{A} += 1;
        $availableTypes{ $type } = 1;
    }
	
	my $tmp = scalar keys %availableTypes;
	#print STDERR "Nb types: $tmp\n";
	#print Dumper(\%typeSize);
	# exit 1;

    my %splitTree = doEvenSplit_rec(\%typeSize, $logNumFolds, \%availableTypes);
	
    my %typeToFold = ();
    doEvenSplit_assignFolds(\%splitTree, \%typeToFold, 0);
    # print Dumper(\%typeToFold);
	
    writeFile($outname, \%typeToFold);
	
	my %foldInfo = ();
    for (my $n=0; $n<$N; $n++) {
        my $type = $allData[$n]{'phrase'};
        if (not defined $typeToFold{$type}) { die "type $type did not get a fold!"; }
        my $f = $typeToFold{$type};

        $allData[$n]{'devfold'}  = $f;
        $allData[$n]{'testfold'} = ($f+1) % $numFolds;
#        $allData[$n]{'testfold'}  = $f;
        $foldInfo{$f}{N} += ( $allData[$n]{'label'} > 0 ) ? 0 : 1;
        $foldInfo{$f}{P} += ( $allData[$n]{'label'} > 0 ) ? 1 : 0;
        $foldInfo{$f}{A} += 1;
        $foldInfo{$f}{T}{$type} = 1;
    }
	#print STDERR "fold\tnb_examples\tneg_examples\tpos_examples\tnb types\n";
    #foreach my $f (sort { $a <=> $b } keys %foldInfo) {
    #    print STDERR "$f:\t" . (join "\t", ($foldInfo{$f}{A}, $foldInfo{$f}{N}/$foldInfo{$f}{A}, $foldInfo{$f}{P}/$foldInfo{$f}{A}, scalar keys %{$foldInfo{$f}{T}})) . "\n";
    #}
}

sub doEvenSplit_assignFolds {
    my ($tree, $typeToFold, $curFold) = @_;
    if (defined $tree->{TYPES}) {
        foreach my $type (keys %{$tree->{TYPES}}) {
            $typeToFold->{$type} = $curFold;
        }
        return $curFold+1;
    }
    $curFold = doEvenSplit_assignFolds(\%{$tree->{LEFT }}, $typeToFold, $curFold);
    $curFold = doEvenSplit_assignFolds(\%{$tree->{RIGHT}}, $typeToFold, $curFold);
    return $curFold;

}

sub writeFile {
	my ($outname, $typeToFold) = @_;
	open O, "> $outname" or die $!;
	
	foreach my $type (keys %{$typeToFold}) {
		print O $type . "\t" . $typeToFold->{$type} . "\n";
	}
	
	close O;
}

sub doEvenSplit_rec {
    my ($typeSize, $splitsToGo, $availableTypes) = @_;
    my %this = ();
    if (($splitsToGo <= 0) || (scalar keys %$availableTypes < 2)) {
        %{$this{TYPES}} = %$availableTypes;
        return (%this);
    }

=pod
    my %side = ();
    {
        my ($t0) = sort { $typeSize->{$a}{A} <=> $typeSize->{$b}{A} } keys %$availableTypes;
        delete $availableTypes->{$t0};
        push @{$side{0}}, $t0;
        @{$side{1}} = ();
    }

    while (scalar keys %$availableTypes > 0) {
        my $bestScore = 0;
        my $bestType  = '';
        my $bestSide  = '';
        foreach my $s (0, 1) {
            foreach my $t (keys %$availableTypes) {
                push @{$side{$s}}, $t;
                my $score = evenSplitQuality($typeSize, \@{$side{0}}, \@{$side{1}});
                if (($bestType eq '') || ($score > $bestScore)) {
                    $bestScore = $score;
                    $bestType  = $t;
                    $bestSide  = $s;
                }
                pop @{$side{$s}};
            }
        }
        print STDERR "bestScore = $bestScore, bestType = $bestType, bestSide = $bestSide\n";
        push @{$side{$bestSide}}, $bestType;
        delete $availableTypes->{$bestType};
    }
=cut


    my @nextTypes = sort { $typeSize->{$a}{A} <=> $typeSize->{$b}{A} } keys %$availableTypes;
	
	#my $tmp = scalar @nextTypes;
	#print STDERR "nb types: $tmp\n";
	#print Dumper(\@nextTypes);
	#exit 1;
    
	my %side = ();
    @{$side{0}} = ();    @{$side{1}} = ();
    {
        my $t = pop @nextTypes;
        push @{$side{0}}, $t;
    }

    while (scalar @nextTypes > 0) {
        my $t = pop @nextTypes;
        my $bestScore = 0;
        my $bestSide  = '';
        foreach my $s (0, 1) {
            push @{$side{$s}}, $t;
            my $score = evenSplitQuality($typeSize, \@{$side{0}}, \@{$side{1}});
            if (($bestSide eq '') || ($score > $bestScore)) {
                $bestScore = $score;
                $bestSide  = $s;
            }
            pop @{$side{$s}};
        }
        # print STDERR "word = $t, bestScore = $bestScore, bestSide = $bestSide\n";
        push @{$side{$bestSide}}, $t;
    }
	


=pod
    my %side = ();
    {
        my $t0 = popRandomKey($availableTypes);
        push @{$side{0}}, $t0;
        @{$side{1}} = ();
    }

    my $s = 1;
    while (scalar keys %$availableTypes > 0) {
        my $bestScore = 0;
        my $bestType  = '';
        foreach my $t (keys %$availableTypes) {
            push @{$side{$s}}, $t;
            my $score = evenSplitQuality($typeSize, \@{$side{0}}, \@{$side{1}});
            if (($bestType eq '') || ($score > $bestScore)) {
                $bestScore = $score;
                $bestType  = $t;
            }
            pop @{$side{$s}};
        }
        print STDERR "bestScore = $bestScore, bestType = $bestType\n";
        push @{$side{$s}}, $bestType;
        delete $availableTypes->{$bestType};
        $s = 1-$s;
    }
=cut

    my %left = (); foreach my $x (@{$side{0}}) { $left{$x} = 1; }
    my %right = (); foreach my $x (@{$side{1}}) { $right{$x} = 1; }
    %{$this{LEFT}}  = doEvenSplit_rec($typeSize, $splitsToGo-1, \%left);
    %{$this{RIGHT}} = doEvenSplit_rec($typeSize, $splitsToGo-1, \%right);
    return (%this);
}

sub popRandomKey {
    my ($h) = @_;
    my @k = keys %$h;
    if (@k == 0) { return undef; }
    my $i = int(rand() * scalar @k);
    delete $h->{$k[$i]};
    return $k[$i];
}

sub evenSplitQuality {
    my ($typeSize, $leftIDs, $rightIDs) = @_;

    my %leftInfo = (N => 0, P => 0, A => 0);
    foreach my $type (@$leftIDs) {
        $leftInfo{N} += $typeSize->{$type}{N};
        $leftInfo{P} += $typeSize->{$type}{P};
        $leftInfo{A} += $typeSize->{$type}{A};
    }
    $leftInfo{N} /= $leftInfo{A} if $leftInfo{A} > 0;
    $leftInfo{P} /= $leftInfo{A} if $leftInfo{A} > 0;

    my %rightInfo = (N => 0, P => 0, A => 0);
    foreach my $type (@$rightIDs) {
        $rightInfo{N} += $typeSize->{$type}{N};
        $rightInfo{P} += $typeSize->{$type}{P};
        $rightInfo{A} += $typeSize->{$type}{A};
    }
    $rightInfo{N} /= $rightInfo{A} if $rightInfo{A} > 0;
    $rightInfo{P} /= $rightInfo{A} if $rightInfo{A} > 0;

    my $nAvg = ($leftInfo{N} + $rightInfo{N}) / 2;
    my $pAvg = ($leftInfo{P} + $rightInfo{P}) / 2;

    my $klLeft  = (($leftInfo{N} <= 0) ? 0 : ( $leftInfo{N} * log0( $leftInfo{N} / $nAvg ) )) +
                  (($leftInfo{P} <= 0) ? 0 : ( $leftInfo{P} * log0( $leftInfo{P} / $pAvg ) ));
    my $klRight = (($rightInfo{N} <= 0) ? 0 : ( $rightInfo{N} * log0( $rightInfo{N} / $nAvg ) )) +
                  (($rightInfo{P} <= 0) ? 0 : ( $rightInfo{P} * log0( $rightInfo{P} / $pAvg ) ));

    my $js = ( $klLeft + $klRight ) / 2;
    my $sizeDiff = abs($leftInfo{A} - $rightInfo{A});

#    print STDERR "> $nAvg $pAvg $klLeft $klRight | $leftInfo{N} $leftInfo{P} $rightInfo{N} $rightInfo{P} | " . 
#        (join ' : ', ( $leftInfo{N}/($nAvg+0.0001) , $leftInfo{P} / ($pAvg+0.0001), $rightInfo{N} / ($nAvg+0.0001), $rightInfo{P} / ($pAvg+0.0001) ));
#    print STDERR "\n> js = $js, sizeDiff = $sizeDiff -> " . (exp(-$sizeDiff / scalar keys %$typeSize)) . "\n";
    return (-10*$js - ($sizeDiff / scalar keys %$typeSize)/10000);
}

sub doUnevenSplit {
    # assign data points to folds
    my %allPhrases = ();
    for (my $n=0; $n<$N; $n++) {
        $allPhrases{  $allData[$n]{'phrase'}  } = -1;
    }

    my @allPhrases = keys %allPhrases;
    my @fold = ();
    for (my $i=0; $i<@allPhrases; $i++) {
        $fold[$i] = $i % $numFolds;
    }
    for (my $i=0; $i<@allPhrases; $i++) {
        my $j = int($i + rand() * (@allPhrases - $i));
        my $t = $fold[$i];
        $fold[$i] = $fold[$j];
        $fold[$j] = $t;

        $allPhrases{ $allPhrases[$i] } = $fold[$i];
    }

    for (my $n=0; $n<$N; $n++) {
        $allData[$n]{'testfold'} = $allPhrases{ $allData[$n]{'phrase'} };
        $allData[$n]{'devfold' } = (1+$allPhrases{ $allData[$n]{'phrase'} }) % $numFolds;
    }
}
