from calcom.io import CCConverter

class CCBioConverter(CCConverter):
    def __init__(self, defaultOutType=None, defaultInType=None):
        super().__init__(defaultOutType=defaultOutType, defaultInType=defaultInType)

    def convVariableName(self, variable_name, omicsType, omicsFormat, outType):
        return variable_name # TODO this does nothing

    def convert(self, ccd, inType=None, outType=None):
        if not inType:
            inType = self.defaultInType if self.defaultInType else [None, None]
            omicsType = 'omics_type'
            if omicsType in ccd.study_attrs:
                inType[0] = ccd.study_attrs[omicsType]

            omicsFormat = 'omics_format'
            if omicsFormat in ccd.study_attrs:
                inType[1] = ccd.study_attrs[omicsFormat]

            if inType[0] == None or inType[1] == None:
                return ccd

        if not outType:
            if self.defaultOutType:
                outType = self.defaultOutType
            else:
                return ccd

        # POST: inType: [omicsType, omicsFormat]
        # POST: outType: TODO

        # Do conversion
        for i,e in enumerate(ccd.variable_names):
            ccd.variable_names[i] = self.convVariableName(e, inType[0], inType[1], outType)
            # TODO outType needs to be adapted when it's defined.

        return ccd
